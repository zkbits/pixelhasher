#include "json.h"

// ghetto lib: copy json value specified by `key` to `dst`
//
// can get a json (string or integer) value as a string.
bool json_get(char *json, char *key, char *dst) {
  char keybuf[1024];
  char *final_p = json + strlen(json) - 1;
  snprintf(keybuf, 1024, "\"%s\"", key);
  char *p = strstr(json, keybuf);
  if (p == NULL) return false;
  p = strstr(p + strlen(keybuf), ":");
  if (p == NULL) return false;
  p++;
  while (p[0] == ' ' || p[0] == '\n' || p[0] == '\t') { p++; if (p == final_p) return false; }
  if (p[0] == '"') {
    p++;
    if (p == final_p) return false;
    char *p2 = strstr(p, "\"");
    if (p2 == NULL) return false;
    strncpy(dst, p, p2 - p);
    dst[p2 - p] = '\0';
    return true;
  }
  else if (p[0] >= '0' && p[0] <= '9') {
    char *p2 = dst;
    while (p[0] >= '0' && p[0] <= '9') { p2[0] = p[0]; p2++; p++; }
    p2[0] = '\0';
    return true;
  }
  return false;
}
