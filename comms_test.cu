#include "comms.h"
#include "misc.h"

void print_work(work_t w) {
  printf("mining_target=0x"); print_bytes(w.mining_target, 32);
  printf(" miner_address=0x"); print_bytes(w.miner_address, 20);
  printf(" challenge_number=0x"); print_bytes(w.challenge_number, 32);
  printf(" sprite=0x"); print_bytes(w.sprite, 32);
}

int main(int argc, char **argv)
{
  config_t config;
  work_t current_work_x;
  sprintf(config.server_name, "127.0.0.1");
  config.server_port = 17890;
  sprintf(config.payout_address, "%s", argv[1]);

  printf("Pool server is %s:%d\n", config.server_name, config.server_port);
  printf("Payout address is %s\n", config.payout_address);


  printf("Start connection thread\n");
  start_comms_thread(config);
  printf("Start main GPU thread\n");

  char *solution = "0xSOLN";

  bool sent = false;
  while (true) {
    comms_copy_current_work(&current_work_x);
    printf("mining_target="); print_work(current_work_x); printf(")\n");
    sleep(1); /// do the handoff to the gpu, run 4 billion times
    if (!sent) {
      comms_submit_solution((char*)solution);
      sent = true;
    }
  }

  join_comms_thread();
}

