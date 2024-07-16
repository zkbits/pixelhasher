#include "json.h"
#include <assert.h>

void test1() {
  char *json = (char*)R"({"version": 0, "payout_address": "0x999", "bar": 989898x9})";
  char value[1024];
  bool b;
  b = json_get(json, (char*)"version", value);
  assert(b == true);
  assert(strcmp("0", value) == 0);
  b = json_get(json, (char*)"bar", value);
  assert(b == true);
  b = json_get(json, (char*)"payout_address", value);
  assert(b == true);
  assert(strcmp("0x999", value) == 0);
}

void test2() {
  char value[1024];
  char *json = (char*)"{\"version\": 0,\n   \"payout_address\": \n\n\"0x999\", \"bar\": 98989\"";
  bool b;
  b = json_get(json, (char*)"payout_address", value);
  assert(b == true);
  assert(strcmp("0x999", value) == 0);
  b = json_get(json, (char*)"zzzzzzzzzzzz", value);
  assert(b == false);
}
int main(int argc, char **argv) {
  test1();
  test2();
  printf("tests passed\n");
  return 0;
}
