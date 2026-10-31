#ifndef OCRUS_H
#define OCRUS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcrusEngine OcrusEngine;
typedef struct OcrusCancel OcrusCancel;

/* ABI 1. All functions run in the caller's process. */
uint32_t ocrus_abi_version(void);

/* Status: 0 success, 1 invalid input, 2 engine error, 3 cancelled, 4 panic.
 * mode: 0 auto, 1 accurate. threads must be between 1 and 64.
 * Output slots must initially be NULL. Errors and JSON are UTF-8 strings owned
 * by Rust; release each non-NULL string with ocrus_string_free, never free().
 * Handles must remain alive until all calls using them have returned.
 * Recognize calls on the same engine must not overlap; cancellation may run on
 * another thread. After a panic, discard the engine and create a new one. */
int32_t ocrus_engine_create(const char *model_dir, uint32_t mode,
                          uint32_t threads, OcrusEngine **out_engine,
                          char **out_error);
void ocrus_engine_free(OcrusEngine *engine);

OcrusCancel *ocrus_cancel_create(void);
void ocrus_cancel_request(const OcrusCancel *cancel);
void ocrus_cancel_free(OcrusCancel *cancel);

/* Pixels: top row first, RGBA8 (format 0) or BGRA8 (format 1).
 * stride includes row padding. len must cover stride * height bytes.
 * Cancellation is checked before preparation, between model runs and on return.
 * Recognition never writes the input buffer and retains no pointer to it. */
int32_t ocrus_recognize_pixels(const OcrusEngine *engine,
                             const uint8_t *pixels, size_t len,
                             uint32_t width, uint32_t height, size_t stride,
                             uint32_t format, const OcrusCancel *cancel,
                             char **out_json, char **out_error);
void ocrus_string_free(char *string);

#ifdef __cplusplus
}
#endif
#endif
