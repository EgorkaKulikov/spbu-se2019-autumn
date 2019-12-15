using System;
using System.Threading.Tasks;

namespace Task02
{
    public class Sorts
    {
        private static Graph.Edge[] _edgesArray;

        public static void Swap(long i, long j)
        {
            // Swap two element in an _edgesArray with given indexes.
            var temp = _edgesArray[i];
            _edgesArray[i] = _edgesArray[j];
            _edgesArray[j] = temp;
        }
        static long Partition(long low, long high)
        {
            var pivot = _edgesArray[high];

            // Index of smaller element 
            long i = (low - 1);

            for (long j = low; j <= high - 1; j++)
            {
                // If current element is smaller 
                // than or equal to pivot 
                if (_edgesArray[j].CompareTo(pivot) < 0)
                {
                    i++; // increment index of 
                    // smaller element 
                    Swap(i, j);
                }
            }

            Swap(i + 1, high);
            return (i + 1);
        }
        public static void QuickSortIterative(long lo, long hi)
        {
            if (lo >= hi) return;
            long[] stack = new long[4 * hi];
            long top = 0;
            stack[top++] = lo;
            stack[top++] = hi;
            while (top > 0)
            {
                hi = stack[--top];
                lo = stack[--top];
                if (lo >= hi)
                {
                    continue;
                }

                long p = Partition(lo, hi);
                stack[top++] = lo;
                stack[top++] = p - 1;
                stack[top++] = p + 1;
                stack[top++] = hi;
            }
        }


        public static void QuickSortParallel(long border, long lo, long hi)
        {
            if (hi - lo < border)
            {
                QuickSortIterative(lo, hi);
            }
            else
            {
                long p = Partition(lo, hi);
                Parallel.Invoke(
                    () => QuickSortParallel(border, lo, p - 1),
                    () => QuickSortParallel(border,p + 1, hi)
                );
            }
        }

        public static void StartQuickSortParallel(long verticesNumber, ref Graph.Edge[] edges)
        {
            _edgesArray = edges;
            QuickSortParallel(edges.Length / Environment.ProcessorCount, 0, edges.Length - 1);
            edges = _edgesArray;
        }

        public static void StartQuickSortSeq(ref Graph.Edge[] edges)
        {
            _edgesArray = edges;
            QuickSortIterative(0, edges.Length - 1);
            edges = _edgesArray;
        }
    }
}