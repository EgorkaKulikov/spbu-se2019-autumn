using System;

namespace Graphs
{
    public class SeqSort
    {
        private static readonly Random Random = new Random();
         
        public static void StartQuickSort<T>(T[] items) where T : IComparable<T>
        {
            QuickSort(items, 0, items.Length);
        }
         
        internal static void QuickSort<T>(T[] items, int left, int right)
            where T : IComparable<T>
        {
            if (right - left < 2) return;
            int pivot = Partition(items, left, right);
            QuickSort(items, left, pivot);
            QuickSort(items, pivot + 1, right);
        }
         
        internal static int Partition<T>(T[] items, int left, int right)
            where T : IComparable<T>
        {
            int pivotPos = Random.Next(left, right);
            T pivotValue = items[pivotPos];
 
            Swap(ref items[right - 1], ref items[pivotPos]);
            int store = left;
 
            for (int index = left; index < right - 1; ++index)
            {
                if (items[index].CompareTo(pivotValue) < 0)
                {
                    Swap(ref items[index], ref items[store]);
                    ++store;
                }
            }
 
            Swap(ref items[right - 1], ref items[store]);
            return store;
        }
         
        private static void Swap<T>(ref T firstElem, ref T secondElem)
        {
            T temp = firstElem;
            firstElem = secondElem;
            secondElem = temp;
        }
    }   
}