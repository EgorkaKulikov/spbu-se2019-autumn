using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Task_02
{
    internal static class KruskalAlgo
    {
        
        public static int KruskalFindMstConsistent(List<Edge> graph, int vertex)
        {
            var MST = new List<Edge>();
            var treeID = new int[vertex];
            var newGraph = graph;
            var MstWeight = 0;
            newGraph.Sort();
            for (var i = 0; i < vertex; i++)
            {
                treeID[i] = i;
            }
            foreach (var edge in newGraph.Where(edge => treeID[edge.From] != treeID[edge.To]))
            {
                
                MST.Add(edge);
                MstWeight += edge.Weight;
                var oldId = treeID[edge.To];  
                var newId = treeID[edge.From];
                for (var j = 0; j < vertex; ++j)
                    if (treeID[j] == oldId)
                        treeID[j] = newId;
            }
            return MstWeight;
        }

        public static int KruskalFindMstParallel(List<Edge> graph, int vertex)
        {
            var MST = new List<Edge>();
            var treeID = new int[vertex];
            for (var i = 0; i < vertex; i++)
            {
                treeID[i] = i;
            }
            var newGraph = graph.ToArray();
            var MstWeight = 0;
            ParallelQuickSort(newGraph, 0, newGraph.Length - 1);
            foreach (var edge in newGraph.Where(edge => treeID[edge.From] != treeID[edge.To]))
            {
                
                MST.Add(edge);
                MstWeight += edge.Weight;
                var oldId = treeID[edge.To];  
                var newId = treeID[edge.From];
                for (var j = 0; j < vertex; ++j)
                    if (treeID[j] == oldId)
                        treeID[j] = newId;
            }
            return MstWeight;
        }
        private static void ParallelQuickSort(Edge[] arr, int left, int right)
        {
            if (right - left < 2) return;
            var q = Partition(arr, left, right);
            // 512 - some magic number
            if (right - left > 512)
            {
                Parallel.Invoke(
                    () => ParallelQuickSort(arr, left, q),
                        () => ParallelQuickSort(arr, q + 1, right)
                );
            }
            else
            {
                ConsistentQuickSort(arr, left, q);
                ConsistentQuickSort(arr, q + 1, right);
            }
            
        }

        private static void ConsistentQuickSort(Edge[] arr, int left, int right)
        {
            if (left >= right) return;
            var q = Partition(arr, left, right);
            ConsistentQuickSort(arr, left, q);
            ConsistentQuickSort(arr, q + 1, right);
        }

        private static int Partition(Edge[] arr, int left, int right)
        {
            var middle = arr[(left + right) / 2].Weight;
            var i = left;
            var j = right;
            while (i <= j)
            {
                while (arr[i].Weight < middle)
                    i++;
                while (arr[j].Weight > middle)
                    j--;
                if (i >= j)
                    break;
                Swap(ref arr[i++], ref arr[j--]);
            }
            return j;
        }

        private static void Swap(ref Edge first, ref Edge second)
        {
            var temp = first;
            first = second;
            second = temp;
        }
    }
}