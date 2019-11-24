#ifndef __TEST_H_
#define __TEST_H_

#define mu_assert(message, test) do { if (!(test)) return message; } while (0)
 #define mu_run_test(test) do { char *message = test(n, a, b); tests_run++; \
                                if (message) return message; } while (0)
 extern int tests_run;
 
 int run_tests(int n, double *a, double *b);
 
 #endif