#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <ctype.h>
#include <sys/time.h>
#include <unistd.h>
#include "comms.h"
#include "config.h"
#include "keccak2.h"
#include "len.h"
#include "misc.h"

void gpu_init();

#define NUMBER_BLOCKS 256
#define NUMBER_THREADS 256
#define GPU_OUTER_LOOP_COUNT 256 // GOLC entropy is log2(GOLC), need 8 bits!
#define GPU_INNER_LOOP_COUNT 256 // GILC entropy is log2(GILC), need 8 bits!

work_t current_work, last_work;

cudaEvent_t start, stop;

// all optimizations should come out of this 'cuz it only hits 1 or a few times
// per solution found. yet taking out the pragma slows it down??
__device__ int compare_hash(uint8_t *target, uint8_t *hash) {
// seems to speed up ~8% ?? why
#pragma unroll 16
  for (size_t i = 0; i < 32; i++) {
    if (hash[i] > target[i]) {
      return 0;
    }
    if (target[i] > hash[i]) {
      return 1;
    }
  }
  return 1;
}

__device__ void keccak(const uint8_t *initial_hash_input, uint8_t *winning_hash_input, int *done, int tid, const uint8_t *mining_target)
{
  uint64_t state[25];
  uint8_t temp[144];
  int rsize = 136;
  int rsize_byte = 17;

  // pad the string out according to keccak spec
  memset(temp, 0, sizeof temp);
  memcpy(temp, initial_hash_input, LEN84_INPUT);
  temp[LEN84_INPUT] = 1;
  memset(temp + LEN84_INPUT+1, 0, rsize - (LEN84_INPUT+1));
  temp[rsize - 1] |= 0x80;

  // 16 bits entropy to zkbit first row  left, right sides
  temp[LEN32_CHNUM + LEN20_SENDR + 0] = tid;
  temp[LEN32_CHNUM + LEN20_SENDR + 1] = tid>>8;

  for (int j = 0; j < GPU_OUTER_LOOP_COUNT; j++) {
    temp[LEN84_INPUT - 1] = j; // 8 bits entropy to zkbit last row (right side)
    for (int il = 0; il < GPU_INNER_LOOP_COUNT; il++) {
      temp[LEN84_INPUT - 2] = il; // 8 bits entropy to last row (left side)

      // note: could instead zero state[i] from rsize_byte-1 to 24,
      // then = (instead of ^=) into state[i], but i found no speedup
      for (int i = 0; i < 25; i++)
        state[i] = 0;
      for (int i = 0; i < rsize_byte; i++) {
        state[i] ^= ((uint64_t *) temp)[i];  // ^= is small speedup (somehow)
      }

      keccak256(state);
      if ((uint32_t)state[0]) continue; // micro speedup: only 1 in 2**32 pass

      if(compare_hash((uint8_t*)mining_target, (uint8_t*)state)) {
        done[0] = 1;
        /*
        printf("found solution in tid=%d(0x%0x) at j=%d(0x%0x), il=%d(0x%0x)\n",tid,tid,j,j,il,il);
        printf("  solution:\n");
        printf("    ");
        for (int i = 0; i < LEN32_NONCE; i++) {
          printf("%02x", (uint8_t)temp[i+52]);
        }
        printf("\n");
        printf("  full hash input:\n");
        printf("    challenge.......................................................minerEthAddress.........................solution........................................................\n");
        printf("    ");
        for (int i = 0; i < 84; i++) {
          printf("%02x", (uint8_t)temp[i]);
        }
        printf("\n");


        printf("  hash of full proof (must be less than mining_target):\n");
        printf("    ");
        for (int i = 0; i < LEN32_KSHA3; i++) {
          printf("%02x", ((uint8_t*)state)[i]);
        }
        printf("\n");
        */
        memcpy(winning_hash_input, temp, LEN84_INPUT);
        //break;
      }
    }
  }
}

__global__ void launch(uint8_t *mining_target, uint8_t *initial_hash_input, uint8_t *winning_hash_input, int *done)
{
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  keccak(initial_hash_input, winning_hash_input, done, tid, mining_target);
}

void gpu_init(char *gpuDescription)
{
  cudaDeviceProp device_prop;
  int device_count;

  cudaGetDeviceCount(&device_count);
  if (device_count != 1) {
    printf("\nHelp: Select a GPU like this:\n\n");
    printf("  CUDA_VISIBLE_DEVICES=n ./pixelhasher\n");
    printf("                       ^ put GPU index here (0 is first GPU)\n\n");
    exit(EXIT_FAILURE);
  }

  if (cudaGetDeviceProperties(&device_prop, 0) != cudaSuccess) {
    printf("Problem getting properties for device, exiting...\n");
    exit(EXIT_FAILURE);
  }

  printf("name:        %s\n", device_prop.name);
  printf("pciBusID:    %d\n", device_prop.pciBusID);
  printf("pciDeviceID: %d\n", device_prop.pciDeviceID);
  // todo in future: calculate more optimal (threads x blocks) from device_prop? micro speedup only

  sprintf(gpuDescription, "%s, PCI:%d:%d", device_prop.name, device_prop.pciBusID, device_prop.pciDeviceID);
}

double now() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec) + (double)(tv.tv_usec) / 1000000 ;
}

void seed() {
  FILE *randomData = fopen("/dev/urandom", "r");
  if (randomData == NULL) {
    printf("bad random 1\n");
    exit(1);
  }
  unsigned int s;
  ssize_t result = fread(&s, sizeof s, 1, randomData);
  if (result < 0) {
    printf("bad random 2\n");
    exit(1);
  }
  printf("seed=%u\n", s);
  srand(s);
}

// mangle the initial_hash_input until keccak(hash_input) < mining_target
// returns:
//  1  -> success. initial_hash_input is now well mangled (solved), copied to
//  winning_hash_input. can extract solution nonce from last 64 bytes of it
//  0 -> detected that new contract has hit, so abandoned work
//  -1 -> some cuda error
int solve_work(uint8_t *full_solution) {
  float h_to_d_time = 0.0;
  float comp_time = 0.0;
  float d_to_h_time = 0.0;
  float total_time = 0.0;
  size_t digest_str_size = LEN32_KSHA3;
  double last_work_check = now();
  double last_printout = now();
  double started_at = now();

  uint8_t initial_hash_input[LEN84_INPUT];
  copy_bytes(initial_hash_input, current_work.challenge_number, LEN32_CHNUM);
  copy_bytes(initial_hash_input + 32, current_work.miner_address, LEN20_SENDR);

  uint8_t h_mining_target[LEN32_KSHA3];
  memcpy(h_mining_target, current_work.mining_target, LEN32_MTRGT);
  print_bytes2("h_mining_target=", h_mining_target, LEN32_MTRGT);

  int h_done[1] = {0};
  uint64_t starting_tid = 0;

  int *d_done;
  uint8_t *d_mining_target;
  uint8_t *d_winning_hash_input;
  uint8_t *d_initial_hash_input;

  cudaMalloc((void**) &d_done, sizeof(int));
  cudaMalloc((void**) &d_mining_target, digest_str_size);
  cudaMalloc((void**) &d_winning_hash_input, LEN84_INPUT);
  cudaMalloc((void**) &d_initial_hash_input, LEN84_INPUT);

  cudaEventRecord(start, 0);
  cudaMemcpy(d_done, h_done, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mining_target, h_mining_target, LEN32_KSHA3, cudaMemcpyHostToDevice);
  cudaMemcpy(d_initial_hash_input, initial_hash_input, LEN84_INPUT, cudaMemcpyHostToDevice);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&h_to_d_time, start, stop);

  cudaMemcpy(d_winning_hash_input, "", 1, cudaMemcpyHostToDevice);
  cudaEventRecord(start, 0);
  printf("threads=%d, blocks=%d\n", NUMBER_BLOCKS, NUMBER_THREADS);
  int count = 0;
  cudaError_t cudaerr;

  double runtime;
  while (true) {
    // art mode
    memcpy(initial_hash_input + LEN32_CHNUM + LEN20_SENDR, current_work.sprite, LEN32_NONCE);

    printf("\nInjecting entropy\n");
    // put entropy in left and right borders.
    // if the gpu kernel is grinding thru 32 effective bits of its own entropy
    // too, then we have 60 bits total
    char v;
    int o;
    for (int i = 1; i < 15; i++) {
      o = 52 + i * 2;
      v = initial_hash_input[o];
      v &= 0x7f;
      if (rand() % 2 == 1) {
        v |= 0x80;
      }
      initial_hash_input[o] = v;

      o = 52 + i * 2 + 1;
      v = initial_hash_input[o];
      v &= 0xfe;
      if (rand() % 2 == 1) {
        v |= 0x1;
      }
      initial_hash_input[o] = v;
    }

    print_bytes2("initial_hash_input[52..83]=", initial_hash_input + 52, 32);

    printf("vvvvvvvvvvvvvvvv\n");
    print_art32(initial_hash_input + 52);
    printf("^^^^^^^^^^^^^^^^\n");

    cudaStream_t d;
    cudaStreamCreate(&d);
    cudaMemcpy(d_initial_hash_input, initial_hash_input, LEN84_INPUT, cudaMemcpyHostToDevice);
    runtime = now();
    launch<<<NUMBER_BLOCKS, NUMBER_THREADS, 0, d>>>(d_mining_target, d_initial_hash_input, d_winning_hash_input, d_done);
    starting_tid += NUMBER_BLOCKS * NUMBER_THREADS;
    while (cudaStreamQuery(d) != cudaSuccess) { usleep(50000); }
    cudaerr  = cudaDeviceSynchronize();
    runtime = now() - runtime;
    cudaMemcpy(h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);

    if (cudaerr != cudaSuccess) {
      printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
      return -1;
    }
    if (h_done[0])
      break; // found solution
    if(true) {
      double n = now();
      if(n - last_printout > 1) {
        double elapsed = n - started_at;
        last_printout = n;
        uint64_t computations = starting_tid*GPU_INNER_LOOP_COUNT*GPU_OUTER_LOOP_COUNT;
        double cps = (double)computations / elapsed;
        fprintf(stderr, "\033[1K\relapsed: %ds / count: %d / computations: %lu / speed: %lu/s / runtime: %f", (int)elapsed, count, computations, (uint64_t)cps, runtime);
      }
      if(n - last_work_check > 60) {
        last_work_check = n;
        work_t new_work_x;
        comms_copy_current_work(&new_work_x);
        if (!work_eql(new_work_x, current_work)) {
          printf("\n");
          printf("solve_work: Bail because there is new work detected\n");
          return 0;
        }
      }
    }
    count++;
  }
  printf("\n");
  printf("count = %d\n", count);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&comp_time, start, stop);

  cudaEventRecord(start, 0);
  cudaMemcpy(full_solution, d_winning_hash_input, LEN84_INPUT, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&d_to_h_time, start, stop);

  h_to_d_time /= 1000;
  comp_time /= 1000;
  d_to_h_time /= 1000;
  total_time = h_to_d_time + comp_time + d_to_h_time;

  uint64_t computations = (uint64_t)starting_tid*GPU_INNER_LOOP_COUNT*GPU_OUTER_LOOP_COUNT;

  print_bytes2("full_solution=", full_solution, LEN84_INPUT);
  printf("Memory transfer (host to device) | %0.6f seconds\n", h_to_d_time);
  printf("     Total time to find solution | %.1f seconds\n", comp_time);
  printf("  Total keccak() hashes computed | %lu\n", computations);
  printf("          keccak() hashes/second | %.0f\n", (float)computations/comp_time);
  printf("Memory transfer (device to host) | %0.6f seconds\n", d_to_h_time);
  printf("                      Total time | %.1f seconds\n", total_time);

  return 1;
}

void sleep_ms(int ms) {
  struct timespec s;
  assert(ms < 1000);
  s.tv_sec  = 0;
  s.tv_nsec = ms * 1000000L;
  nanosleep(&s, NULL);
}

void print_work(work_t w) {
  printf("mining_target=0x"); print_bytes(w.mining_target, LEN32_MTRGT);
  printf(" miner_address=0x"); print_bytes(w.miner_address, LEN20_SENDR);
  printf(" challenge_number=0x"); print_bytes(w.challenge_number, LEN32_CHNUM);
  printf(" sprite=0x"); print_bytes(w.sprite, LEN32_NONCE);
}

void get_work() {
  comms_copy_current_work(&current_work);
}

int main(int argc, char **argv)
{
  config_t config;
  config = load_config();

  char timebuf[26];
  char gpuDescription[128];
  gpu_init(gpuDescription);
  printf("%s\n", gpuDescription);

  printf("Pool server is %s:%d\n", config.server_name, config.server_port);
  printf("Payout address is %s\n", config.payout_address);
  printf("Start connection thread\n");
  start_comms_thread(config);

  write_time(timebuf);

  seed();

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double started_at = now();
  const char *waiter = "|/-\\|/-\\";
  int wait_count = 0;
  while(true) {
    wait_count += 1;
    get_work();
    if(work_eql(current_work, last_work)) {
      printf("\033[1K\rwaiting for work (%c)", waiter[wait_count/2%8]);
      fflush(stdout);
      sleep_ms(100);
      continue;
    }
    printf("\n");
    copy_work(&last_work, &current_work);
    printf("current_work=("); print_work(current_work); printf(")\n");

    uint8_t full_solution[LEN84_INPUT];
    int ret = solve_work(full_solution);
    if (ret == 1) {
      uint8_t solution[LEN32_NONCE], solution_hex[LEN32_NONCE*2+1];

      // TODO: do an assert() that correct challenge_number and mining_target are in full_solution

      memcpy(solution, &full_solution[LEN32_CHNUM + LEN20_SENDR], LEN32_NONCE);
      bytes2hex(solution_hex, solution, LEN32_NONCE);
      solution_hex[LEN32_NONCE*2] = '\0';

      printf("\nSolution as hex:\n\n0x%s\n\n", solution_hex);
      printf("\nSolution as zkbit:\n\n");
      print_art32(solution);
      printf("\n");

      // TODO: are challenge_hex, miner_address_hex old debug code? todo: rm
      uint8_t challenge_hex[LEN32_CHNUM * 2], miner_address_hex[LEN20_SENDR * 2];
      bytes2hex(challenge_hex, current_work.challenge_number, LEN32_CHNUM);
      challenge_hex[LEN32_CHNUM*2] = '\0';
      bytes2hex(miner_address_hex, current_work.miner_address, 20);
      miner_address_hex[LEN20_SENDR*2] = '\0';
      char solution_buf[1024];
      sprintf(solution_buf, "0x");
      bytes2hex((unsigned char*)solution_buf + 2, full_solution, LEN84_INPUT);
      solution_buf[LEN84_INPUT*2 + 2] = '\0';
      printf("Submit solution: '%s'\n", solution_buf);

      comms_submit_solution(solution_buf);
      write_time(timebuf);
      puts(timebuf);
    }
    else if (ret == 0) {
      printf("Miner loop terminated to allow getting new work\n");
      continue;
    }
    else {
      fflush(stdout);
      sleep(1);
      printf("Hmm, ret was %d, stopping\n", ret);
      break;
    }
    fflush(stdout);
    wait_count = 0;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
