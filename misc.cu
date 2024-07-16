#include <ctype.h>
#include <time.h>
#include "misc.h"

bool bytes_eql(uint8_t *b1, uint8_t *b2, size_t n) {
  return memcmp((char*)b1, (char*)b2, n) == 0;
}
void print_bytes(uint8_t *buffer, size_t len) {
  for(int i = 0; i < len; i++) {
    printf("%02x", buffer[i]);
  }
}
void print_bytes2(const char *heading, uint8_t *buffer, size_t len) {
  printf("%s", heading);
  print_bytes(buffer, len);
  printf("\n");
}
void hex2bytes(uint8_t *chrs, const uint8_t *hexstr) {
  printf("UPGRADE TO hex2bytesn\n");
  size_t len = strlen((char *)hexstr);
  if(len % 2 != 0)
    return;
  size_t final_len = len / 2;
  for (size_t i=0, j=0; j<final_len; i+=2, j++) {
    chrs[j] = (tolower(hexstr[i]) % 32 + 9) % 25 * 16 + (tolower(hexstr[i+1]) % 32 + 9) % 25;
  }
}
void hex2bytesn(uint8_t *bytes, const char *hexstr, size_t len) {
  assert(strlen((char*)hexstr) >= len * 2);
  for (size_t i=0, j=0; j<len; i+=2, j++) {
    bytes[j] = (tolower(hexstr[i]) % 32 + 9) % 25 * 16 + (tolower(hexstr[i+1]) % 32 + 9) % 25;
  }
}
void bytes2hex(uint8_t *hexstr, uint8_t *chrs, size_t len) {
  char buf[3];
  for(int i = 0; i < len; i++) {
    sprintf(buf, "%02x", (uint8_t)chrs[i]);
    hexstr[i*2] = buf[0];
    hexstr[i*2+1] = buf[1];
  }
}
void print_art32(uint8_t *art) {
  for (int i = 0; i < 32; i++) {
    char c = art[i];
    int z = 128;
    for (int j = 0; j < 8; j++) {
      if (c & z) {
        printf("▒▒");
      }
      else {
        printf("  ");
      }
      z /= 2;
    }
    if (i % 2 == 1) {
      printf("\n");
    }
  }
}
void copy_bytes(uint8_t *dst, uint8_t *src, size_t n) {
  memcpy((char*)dst, (char*)src, n);
}
void write_time(char *buffer) {
  time_t timer;
  struct tm* tm_info;
  time(&timer);
  tm_info = localtime(&timer);
  strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
  puts(buffer);
}
