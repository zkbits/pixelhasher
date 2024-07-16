#ifndef CONFIG_H
#define CONFIG_H
typedef struct {
  char payout_address[43];
  char server_name[128];
  int server_port;
} config_t;
config_t load_config();
#endif
