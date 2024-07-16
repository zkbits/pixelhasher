// handle pool comms in a separate thread, to keep the miner simple
#include <stdio.h>
#include <sys/select.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <netdb.h>
#include <sys/time.h>
#include <linux/tcp.h>
#include <pthread.h>
#include "config.h"
#include "comms.h"
#include "json.h"
#include "misc.h"

#define BIGBUF 1024
#define PING_SECONDS 20

config_t config;
work_t comms_current_work;
char current_solution_hex[BIGBUF];
pthread_t comms_thread;
pthread_mutex_t current_work_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t current_solution_mutex = PTHREAD_MUTEX_INITIALIZER;

bool send_solution = false;

// main calls this (frequently) for up-to-date work from pool
void comms_copy_current_work(work_t *dst) {
  pthread_mutex_lock(&current_work_mutex);
  memcpy(dst, &comms_current_work, sizeof(comms_current_work));
  pthread_mutex_unlock(&current_work_mutex);
}

// main calls this to send solution to pool
// note: if we are not connected, the solution will be sent on next connect
void comms_submit_solution(char *solution_hex) {
  pthread_mutex_lock(&current_solution_mutex);
  strcpy(current_solution_hex, solution_hex);
  send_solution = true;
  pthread_mutex_unlock(&current_solution_mutex);
}

void start_comms_thread(config_t sconfig) {
  config = sconfig;
  pthread_create(&comms_thread, NULL, *do_comms, (void *)&config);
}

void join_comms_thread() {
  pthread_join(comms_thread, NULL);
}

int create_socket() {
  int sock;
  while (true) {
    if((sock = socket(PF_INET, SOCK_STREAM, 0)) >= 0) {
      return sock;
    }
    printf("ERROR: Could not create socket\n");
    sleep(10);
  }
}

struct sockaddr_in server_address;

void resolve_server() {
  memset(&server_address, 0, sizeof(server_address));
  server_address.sin_family = AF_INET;
  server_address.sin_port = htons(config.server_port);
  struct hostent *he;
  if ((he = gethostbyname(config.server_name)) == NULL) {
    printf("\nERROR: Unable to resolve '%s' to an IP address\n", config.server_name);
    exit(1);
  }
  memcpy(&server_address.sin_addr, he->h_addr_list[0], he->h_length);
  signal(SIGPIPE, SIG_IGN); // don't die on write to closed socket
}

int connect_and_log_in() {
  int sock;
  while (true) {
    printf("Connecting to pool server\n");
    sock = create_socket();
    int cn = connect(sock, (struct sockaddr*)&server_address,
        sizeof(server_address));
    if (cn >= 0) {
      break;
    }
    char msg[123];
    sprintf(msg, "Error %d", errno);
    if(errno == ECONNREFUSED)
      sprintf(msg, "ECONNREFUSED");
    else if(errno == EHOSTUNREACH)
      sprintf(msg, "EHOSTUNREACH");
    printf("%s\n", msg);
    sleep(5);
  }

  printf("Connected to pool server\n");

  fcntl(sock, F_SETFL, O_NONBLOCK);

  // ensure our small json messages always go out in a single packet
  int flag = 1; 
  setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (char *) &flag, sizeof(int));

  char login_string[100];
  sprintf(login_string, "{\"method\": \"log_in\", \"payout_address\": \"%s\"}\n", config.payout_address);
  send(sock, login_string, strlen(login_string), 0);
  printf("Logged in as %s\n", config.payout_address);

  return sock;
}

unsigned long now_() {
  return (unsigned long)time(NULL); 
}

// read work from json. ensure hex string length and format ("0x...")
bool json_get_hex2bytesn(char *json, char *key, unsigned char *dst, const int len_bytes) {
  char value[BIGBUF];
  if (json_get(json, key, value) &&
      value[0] == '0' && value[1] == 'x' &&
      strlen(value) == 2 + len_bytes * 2) {
    hex2bytesn(dst, value + 2, len_bytes);
    return true;
  }
  return false;
}

void set_work(char *json) {
  work_t new_work;
  if (json_get_hex2bytesn(json, (char*)"mining_target", new_work.mining_target, 32) &&
      json_get_hex2bytesn(json, (char*)"pool_address", new_work.miner_address, 20) &&
      json_get_hex2bytesn(json, (char*)"challenge_number", new_work.challenge_number, 32) &&
      json_get_hex2bytesn(json, (char*)"sprite", new_work.sprite, 32)) {
    pthread_mutex_lock(&current_work_mutex);
    memcpy(&comms_current_work, &new_work, sizeof(new_work));
    pthread_mutex_unlock(&current_work_mutex);
    printf("Got work\n");
    return;
  }
  printf("set_work failed\n");
}

void *do_comms(void *ptr) {
  int sock;
  bool send_ping = false;
  int last_ping = now_();

  resolve_server();

  while(true) {
    sock = connect_and_log_in();

    int n = 0;
    char buffer[BIGBUF];
    char *writepos = buffer;
    struct timeval timeout = { 0, 10000 };
    while(true) {
      fd_set readfds, writefds;
      FD_ZERO(&readfds);
      FD_ZERO(&writefds);
      FD_SET(sock, &readfds);
      if(send_solution) {
        // note: if send is not possible at the time main requests sending the
        // solution, then when it becomes possible to send the solution, it
        // will be sent, even if main has changed the solution again.
        FD_SET(sock, &writefds);
        //printf("\nSending solution to pool server ...\n");
      }
      if (now_() - last_ping > PING_SECONDS) {
        FD_SET(sock, &writefds);
        //printf("\nSending ping to pool server ...\n");
        send_ping = true;
        last_ping = now_();
      }

      int rc = select(sock+1, &readfds, &writefds, NULL, &timeout);
      if (rc == 0) {
        continue;
      }
      if(FD_ISSET(sock, &readfds)) {
        // lol, just assume we get it in one packet
        n = recv(sock, writepos, BIGBUF, 0);
        if(n > 0) {
          char *endpos = writepos+n;
          for(char *p=writepos; p<endpos; p++) {
            if(*p=='\n') {
              *p = '\0';
              //printf("\n");
              //printf("Received %d chars from server: '%s'\n", (int)strlen(buffer), buffer);
              if(buffer[0] == '*') {
                printf("\nServer message:\n\n%s\n", &buffer[1]);
              }
              else if(strstr(buffer, "set_work")) {
                set_work(buffer);
              }
              else {
                printf("Can't handle mesage from server '%s'\n", buffer);
              }
              endpos=buffer;
              break;
            }
          }
          writepos = endpos;
        }
        else {
          close(sock);
          if(n == 0) {
            printf("Server went away.\n");
            break;
          }
          else if(n == -1 && errno == EAGAIN) {
            // ignore
          }
          else if(n < 0) {
            printf("ERROR: Connection error\n");
            break;
          }
        }
      }
      if(FD_ISSET(sock, &writefds)) {
        if (send_solution) {
          pthread_mutex_lock(&current_solution_mutex);
          char cs_buf[BIGBUF+100];
          sprintf(cs_buf, "{\"method\": \"submit_solution\", \"solution\": \"%s\"}\n", current_solution_hex);
          pthread_mutex_unlock(&current_solution_mutex);
          n = send(sock, cs_buf, strlen(cs_buf), 0);
          //printf("Sent %d chars of '%s'\n", n, cs_buf);
          if (n > 0) {
            printf("Sent solution to server (%d bytes sent)\n", n);
            send_solution = false;
          }
          else {
            printf("Some problem sending solution to server (n=%d), will try again\n", n);
            sleep(1);
          }
        }
        if (send_ping) {
          char ping_buf[BIGBUF];
          sprintf(ping_buf, "{\"method\": \"ping\"}\n");
          n = send(sock, ping_buf, strlen(ping_buf), 0);
          //printf("Sent %d chars of '%s'\n", n, ping_buf);
          send_ping = false;
        }
      }
    }
    sleep(1);
  }

  return ptr;
}

bool work_eql(work_t w1, work_t w2) {
  return
    bytes_eql(w1.challenge_number, w2.challenge_number, LEN32_CHNUM) &&
    bytes_eql(w1.mining_target, w2.mining_target, LEN32_MTRGT) &&
    bytes_eql(w1.miner_address, w2.miner_address, LEN20_SENDR) &&
    bytes_eql(w1.sprite, w2.sprite, LEN32_NONCE);
}

void copy_work(work_t *dst, work_t *src) {
  copy_bytes(dst->challenge_number, src->challenge_number, LEN32_CHNUM);
  copy_bytes(dst->mining_target, src->mining_target, LEN32_MTRGT);
  copy_bytes(dst->miner_address, src->miner_address, LEN20_SENDR);
  copy_bytes(dst->sprite, src->sprite, LEN32_NONCE);
}
