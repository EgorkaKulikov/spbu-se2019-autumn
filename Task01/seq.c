#include <stdio.h>
#include <stdlib.h>
//#include <time.h>

int main(int argc, char *argv[]) {
    //double start = clock();
    int n;
    scanf("%d", &n);
    double a[n][n + 1], c;
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n + 1; j++)
            scanf("%lf", &a[i][j]);
    }
    for (i = 0; i < n - 1; i++) {
        if (a[i][i] == 0) {
            for (k = i + 1; k < n; k++) {
                if (a[k][i] != 0) {
                    for (j = 0; j < n + 1; j++) {
                        c=a[k][j];
                        a[k][j]=a[i][j];
                        a[i][j]=c;
                    }
                }
            }
        }
        if (a[i][i]!=0) {
            for (k = i + 1; k < n; k++) {
                c=a[k][i];
                for (j = 0; j < n + 1; j++) {
                    a[k][j] -= a[i][j] * c / a[i][i];
                }
            }
        }
    }
    for (i = n - 1; i >= 0; i--) {
        a[i][i]=a[i][n]/a[i][i];
        for(j = i - 1; j >= 0; j--) {
            a[j][n] -= a[j][i] * a[i][i];
        }
    }
    FILE * file = fopen("seq_res", "a");
    if (file != NULL)
    {
        for (i = 0; i < n; i++) {
            if (a[i][i] == -0) a[i][i] = 0;
            fprintf(file, "%g%s", a[i][i], " ");
        }
        fprintf(file, "\n");
        //fprintf(file, "%f\n", (clock() - start) / CLOCKS_PER_SEC);
        fclose(file);
    }
    return 0;
}