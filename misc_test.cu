#include "misc.h"
#include <assert.h>

void test1() {
  char *mt_hex = (char*)"ABCD";
  unsigned char mt[2];
  hex2bytesn(mt, mt_hex, 2);
  assert(mt[0] == 0xab);
  assert(mt[1] == 0xcd);

  printf("print_bytes: mt='0x");
  print_bytes(mt, 2);
  printf("'\n");
}

int main(int argc, char **argv) {
  test1();
  printf("tests passed\n");
  return 0;
}
