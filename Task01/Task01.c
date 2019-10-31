#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>

void mainElement( int k, int N, double matrix[N][N + 1], int ord[] ){
  int i, j, i_max = k, j_max = k;
  double temp;

  for ( i = k; i < N; i++ ){
    for ( j = k; j < N; j++ ){
      if ( fabs( matrix[i_max][j_max] ) < fabs( matrix[i][j] ) ){
        i_max = i;
        j_max = j;
      }
    }
  }

	for ( j = k; j < N + 1; j++ ){
		temp = matrix[k][j];
		matrix[k][j] = matrix[i_max][j];
		matrix[i_max][j] = temp;
	}

	for ( i = 0; i < N; i++ ){
		temp = matrix[i][k];
		matrix[i][k] = matrix[i][j_max];
		matrix[i][j_max] = temp;
	}

	i = ord[k];
	ord[k] = ord[j_max];
	ord[j_max] = i;
}

void gaussLinear( int N, double matrix[N][N+1]){
  double ans[N];
  int ord[N];
  int i, j, k;

  for (i=0; i < N + 1; i++ ){
    ord[i] = i;
  }

  for ( k = 0; k < N; k++ ){
    mainElement( k, N, matrix, ord );
    if ( fabs( matrix[k][k] ) < 0.0001 ){
      printf( "Система не имеет единственного решения" );
      return;
    }
    for ( j = N; j >= k; j-- ){
      matrix[k][j] /= matrix[k][k];
    }
    for ( i = k + 1; i < N; i++ ){
      for ( j = N; j >= k; j-- ){
        matrix[i][j] -= matrix[k][j] * matrix[i][k];
      }
    }
  }

  for ( i = 0; i < N; i++ )
    ans[i] = matrix[i][N];
  for (i = N - 2; i >= 0; i-- ){
    for (j = i + 1; j < N; j++ ){
      ans[i] -= ans[j] * matrix[i][j];
    }
  }
}

void gaussParallel( int N, double matrix[N][N+1]){
  double ans[N];
  int ord[N];
  int i, j, k;
  omp_set_num_threads(omp_get_num_procs());

	#pragma omp parallel for
  for (i=0; i < N + 1; i++ ){
    ord[i] = i;
  }

  for ( k = 0; k < N; k++ ){
    mainElement( k, N, matrix, ord );
    if ( fabs( matrix[k][k] ) < 0.0001 ){
      printf( "Система не имеет единственного решения" );
      return;
    }
    for ( j = N; j >= k; j-- ){
      matrix[k][j] /= matrix[k][k];
    }
    for ( i = k + 1; i < N; i++ ){
      for ( j = N; j >= k; j-- ){
        matrix[i][j] -= matrix[k][j] * matrix[i][k];
      }
    }
  }

  #pragma omp parallel for
  for ( i = 0; i < N; i++ ){
    ans[i] = matrix[i][N];
  }
  for (i = N - 2; i >= 0; i-- ){
    for (j = i + 1; j < N; j++ ){
      ans[i] -= ans[j] * matrix[i][j];
    }
  }
}

int main(){
	int i, j, N, temp;
	FILE *file;
	if(fopen("test.txt","r")){
		file = fopen("test.txt","r");
	}
	else{
		printf("Создайте файл test.txt");
	}
	fscanf(file, "%i", &N);
	double matrix[N][N+1];
	
	for(i=0;i<N;i++){
		for(j=0;j<N+1;j++){
			fscanf(file, "%i ", &temp);
			matrix[i][j] = (double) temp;
		}
	}
	fclose(file);
	
	double totalTime;
	
	clock_t startTime = clock();
	for(i=0;i<20;i++){
		gaussParallel(N, matrix);
	}
	clock_t endTime = clock();
	totalTime = (double) (endTime - startTime) / CLOCKS_PER_SEC / 20;
	printf("Time Parallel: %f\n", totalTime);
	
	startTime = clock();
	for(i=0;i<20;i++){
		gaussLinear(N, matrix);
	}
	endTime = clock();
	totalTime = (double) (endTime - startTime) / CLOCKS_PER_SEC / 20;
	printf("Time Linear: %f\n", totalTime);
	
	return(0);
}
