#include <assert.h>
#include <stdio.h>
#include "config.h"
#include "json.h"

#define CONFIG_FN "config.json"

config_t load_config() {
  config_t config;
  FILE *fp = fopen(CONFIG_FN, "r");
  if (fp == NULL) {
    printf("Can't open " CONFIG_FN "\n");
    exit(1);
  }
  char json[1024];
  int n = fread(json, 1, 1024, fp);
  if (n == 0) {
    printf("Can't read " CONFIG_FN "\n");
    exit(1);
  }
  config.payout_address[0] = '\0';
  config.server_name[0] = '\0';
  config.server_port = 0;

  char buf[1024];
  bool b;
  b = json_get(json, (char*)"payout_address", buf);
  if (!b) {
    printf("You must set payout_address in " CONFIG_FN "\n");
    exit(1);
  }
  if (strlen(buf) != 42) {
    printf("payout_address should be ethereum address like 0x...\n");
    exit(1);
  }
  strncpy(config.payout_address, buf, 43);

  b = json_get(json, (char*)"server_name", buf);
  if (!b) {
    printf("You must set server_name in " CONFIG_FN "\n");
    exit(1);
  }
  assert(strlen(buf) < 128);
  strncpy(config.server_name, buf, 128);

  b = json_get(json, (char*)"server_port", buf);
  if (!b) {
    printf("You must set server_port in " CONFIG_FN "\n");
    exit(1);
  }
  assert(strlen(buf) < 6);
  config.server_port = atoi(buf);

  return config;
}
