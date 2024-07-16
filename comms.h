#include "config.h"
#include "len.h"

typedef struct {
  unsigned char challenge_number[32];
  unsigned char mining_target[32];
  unsigned char miner_address[20];
  unsigned char sprite[32];
} work_t;

void *do_comms(void*);
void comms_copy_current_work(work_t *dst);
void comms_submit_solution(char *solution);
void start_comms_thread(config_t config);
void join_comms_thread();
bool work_eql(work_t w1, work_t w2);
void copy_work(work_t *dst, work_t *src);
