using System;
using System.Threading.Tasks;

namespace Task02
{
    public static class Sort
    {
        private static void Swap<T>(ref T i, ref T j)
        {
            var temp = i;
            i = j;
            j = temp;
        }
        
        private static readonly Random Random = new Random();
        
        private static int Partition<T>(T[] array, int left, int right) where T : IComparable<T>
        {
            int pivot = Random.Next(left, right);
            T pivotValue = array[pivot];
            Swap(ref array[right - 1], ref array[pivot]);
            int newPivot = left;

            for (int i = left; i < right - 1; ++i)
            {
                if (array[i].CompareTo(pivotValue) < 0)
                {
                    Swap(ref array[i], ref array[newPivot]);
                    newPivot++;
                }
            }
            
            Swap(ref array[right - 1], ref array[newPivot]);
            return newPivot;
        }
        
        private static void QSort<T>(T[] array, int left, int right) where T : IComparable<T>
        {
            if (left >= right) return;

            int pivot = Partition(array, left, right);
            
            QSort(array, left, pivot);
            QSort(array, pivot + 1, right);
        }
        
        public static void QSortParallel<T>(T[] array, int left, int right) where T : IComparable<T>
        {
            if (left >= right) return;
            
            int pivot = Partition(array, left, right);

            Task leftTask = Task.Run(() => QSort(array, left, pivot));
            Task rightTask = Task.Run(() => QSort(array, pivot + 1, right));
            Task.WaitAll(leftTask, rightTask);
        }
    }
}
