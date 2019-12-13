#define MAX_DEPTH  16
#define MIN_LENGHT 32

__device__
void selection_sort(int *array, int left, int right) {

  for (int i = left; i <= right; ++i) {

    int minValue = array[i];
    int minId    = i;

    for (int j = i + 1; j <= right; ++j) {

      if (array[j] < minValue) {
      
        minValue = array[j];
        minId    = j;
      }
    }

    if (i != minId) {

      array[minId] = array[i];
      array[i]     = minValue;
    }
  }
}

__global__
void quicksort(int *array, int left, int right, int depth){
  
  if (MAX_DEPTH <= depth || MIN_LENGHT >= right - left) {
  
    selection_sort(array, left, right);
    return;
  }

  cudaStream_t s, s1;

  int* leftPtr  = array + left;
  int* rightPtr = array + right;

  int pivot = array[(left + right) / 2];

  int leftValue, rightValue;

  int newRight, newLeft;

  while (leftPtr <= rightPtr) {
  
    leftValue  = *leftPtr;
    rightValue = *rightPtr;

    while (leftValue < pivot && leftPtr < array + right) {
    
      leftPtr++;
      leftValue = *leftPtr;
    }

    while (rightValue > pivot && rightPtr > array+left) {
    
      rightPtr--;
      rightValue = *rightPtr;
    }

    if (leftPtr <= rightPtr) {
    
      *leftPtr  = rightValue;
      *rightPtr = leftValue;
      
      leftPtr++;
      rightPtr--;
    }
  }

  newRight = rightPtr - array;
  newLeft  = leftPtr - array;

  if (left < rightPtr - array) {
    
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    quicksort<<<1, 1, 0, s>>>(array, left, newRight, depth + 1);
    cudaStreamDestroy(s);
  }

  if (leftPtr - array < right) {
    
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    quicksort<<<1, 1, 0, s1>>>(array, newLeft, right, depth + 1);
    cudaStreamDestroy(s1);
  }
}

extern "C"
void gpu_sort(int *array, int size){

  int* gpuArray;
  int left = 0;
  int right = size - 1;

  cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);
  
  cudaMalloc((void**) &gpuArray, size * sizeof(int));
  cudaMemcpy(gpuArray, array, size * sizeof(int), cudaMemcpyHostToDevice);
  
  quicksort<<<1, 1>>>(gpuArray, left, right, 0);
  cudaDeviceSynchronize();
  
  cudaMemcpy(array, gpuArray, size*sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(gpuArray);
  
  cudaDeviceReset();
}
