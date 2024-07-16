#include <stdio.h>
#include "config.h"

int main(int argc, char **argv) {
  config_t config = load_config();
  printf("Loaded config="
      "(payout_address='%s', server_name='%s', server_port=%d)\n",
      config.payout_address, config.server_name, config.server_port);
}
