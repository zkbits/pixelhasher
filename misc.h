#include <assert.h>
#include <stdint.h>
#include <stdio.h>
void print_bytes(unsigned char *buffer, size_t len);
void print_bytes2(const char *heading, unsigned char *buffer, size_t len);
void hex2bytes(unsigned char *chrs, const unsigned char *hexstr);
void hex2bytesn(unsigned char *bytes, const char *hexstr, size_t len);
void bytes2hex(unsigned char *hexstr, unsigned char *chrs, size_t len);
bool bytes_eql(unsigned char *b1, unsigned char *b2, size_t n);
void print_art32(uint8_t *art);
void copy_bytes(uint8_t *dst, uint8_t *src, size_t n);
void write_time(char *buffer);
