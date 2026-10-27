/* Minimal config.h for building libmp3lame with the Android NDK.
 *
 * LAME's sources include <config.h> unconditionally in a few translation units;
 * upstream generates it with autoconf. Android provides all the POSIX headers
 * below, so the portable subset is enough. Keep this file in the repository:
 * the LAME sources themselves are fetched, not committed.
 */
#ifndef PRISM_LAME_CONFIG_H
#define PRISM_LAME_CONFIG_H

#define HAVE_ERRNO_H 1
#define HAVE_FCNTL_H 1
#define HAVE_LIMITS_H 1
#define HAVE_MEMORY_H 1
#define HAVE_STDINT_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_SYS_STAT_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_UNISTD_H 1
#define STDC_HEADERS 1
#define HAVE_MEMMOVE 1
#define HAVE_MEMCPY 1

/* util.h references this type without defining it (autoconf provides it). */
typedef float ieee754_float32_t;
#define HAVE_STRCHR 1
#define HAVE_STRRCHR 1
#define HAVE_STRTOL 1
#define HAVE_STRERROR 1
#define HAVE_GETTIMEOFDAY 1

/* No iconv, no socket/netscape extensions, no assembly optimisations. */
#define HAVE_ICONV 0
#define HAVE_LANGINFO_CODESET 0

#endif /* PRISM_LAME_CONFIG_H */
