#include "ocrus.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
  assert(ocrus_abi_version() == 1);
  OcrusEngine *engine = NULL;
  char *error = NULL;
  assert(ocrus_engine_create(NULL, 0, 1, &engine, &error) == 1);
  assert(engine == NULL && error != NULL);
  ocrus_string_free(error);
  if (argc != 2) return 0;
  error = NULL;
  int status = ocrus_engine_create(argv[1], 0, 2, &engine, &error);
  if (status != 0) { fprintf(stderr, "%s\n", error); ocrus_string_free(error); return 1; }
  uint8_t pixels[8 * 8 * 4];
  memset(pixels, 255, sizeof(pixels));
  char *json = NULL;
  assert(ocrus_recognize_pixels(engine, pixels, sizeof(pixels), 8, 8, 32, 0, NULL, &json, &error) == 0);
  assert(json != NULL && strstr(json, "pages") != NULL && error == NULL);
  ocrus_string_free(json);
  json = NULL;
  OcrusCancel *cancel = ocrus_cancel_create();
  assert(cancel != NULL);
  ocrus_cancel_request(cancel);
  assert(ocrus_recognize_pixels(engine, pixels, sizeof(pixels), 8, 8, 32, 0, cancel, &json, &error) == 3);
  assert(json == NULL && error != NULL);
  ocrus_string_free(error);
  ocrus_cancel_free(cancel);
  ocrus_engine_free(engine);
  return 0;
}
