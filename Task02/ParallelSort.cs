using System;
using System.Threading.Tasks;

namespace Graphs
{
    public class ParallelSort
    {
        public static void StartQuickSort<T>(T[] items) where T : IComparable<T>
        {
            QuickSort(items, 0, items.Length);
        }
        
        private static void QuickSort<T>(T[] items, int left, int right)
            where T : IComparable<T>
        {
            if (right - left < 2) return;
            
            int pivot = SeqSort.Partition(items, left, right);
            if (right - left > 300)
            {
                Task leftTask = Task.Run(() => QuickSort(items, left, pivot));
                Task rightTask = Task.Run(() => QuickSort(items, pivot + 1, right));

                Task.WaitAll(leftTask, rightTask);
            }
            else
            {
                SeqSort.QuickSort(items, left, pivot);
                SeqSort.QuickSort(items, pivot + 1, right);
            }
        }
    }
}