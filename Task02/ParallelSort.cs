using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Task02
{
    public class ParallelSort
    {
        public static void QuickSort<T>(List<T> items) where T : IComparable<T>
        {
            QuickSort(items, 0, items.Count);
        }
        
        private static void QuickSort<T>(List<T> items, int left, int right)
            where T : IComparable<T>
        {
            if (right - left < 2) return;
            int pivot = Partition(items, left, right);

            var leftTask = Task.Run(() => QuickSort(items, left, pivot));
            var rightTask = Task.Run(() => QuickSort(items, pivot + 1, right));

            Task.WaitAll(leftTask, rightTask);
        }
        
        private static int Partition<T>(List<T> items, int left, int right)
            where T : IComparable<T>
        {
            int pivotPos = (left + right) / 2;
            T pivotValue = items[pivotPos];

            Swap(items, right - 1, pivotPos);
            int tempLeft = left;

            for (int i = left; i < right - 1; ++i)
            {
                if (items[i].CompareTo(pivotValue) < 0)
                {
                    Swap(items, i, tempLeft);
                    tempLeft++;
                }
            }

            Swap(items, right - 1, tempLeft);
            return tempLeft;
        }

        private static void Swap<T>(List<T> list, int indexA, int indexB)
        {
            T tmp = list[indexA];
            list[indexA] = list[indexB];
            list[indexB] = tmp;
        }
    }
}
