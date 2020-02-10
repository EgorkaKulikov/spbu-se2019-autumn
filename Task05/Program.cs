using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

namespace Task05
{
    public class Program
    {
        static readonly int AmountThreads = 1;
        static readonly int MaxNumberVertexes = 100000;
        static readonly int[] CountsRequests = new[] { 1000000, 10, 10, 10, 10, 10, 30, 30, 30, 30, 30, 100, 100, 100, 100, 100, 275, 275, 275,
            275, 275, 800, 800, 800, 800, 800, 2500, 2500, 2500, 2500, 2500, 7000, 7000, 7000, 7000, 7000, 22000, 22000, 22000, 22000,
            22000, 65000, 65000, 65000, 65000, 65000, 200000, 200000, 200000, 200000, 200000, 600000, 600000, 600000, 2000000, 2000000,
            2000000, 5000000, 5000000, 5000000, 10000000, 10000000, 10000000 };

        static public void ExecRequests<K, V, NT, TT>(TT tree, params (K, V?)[] requests)
            where K : IComparable<K>, new()
            where V : struct
            where NT : BinaryNode<K, V>
            where TT : BinaryTree<K, V, NT>
        {
            Stack<Thread> threads = new Stack<Thread>();
            for (int i = 0; i < AmountThreads; i++)
            {
                int ind = i;
                threads.Push(new Thread(() =>
                {
                    for (int j = ind; j < requests.Length; j += AmountThreads)
                        tree[requests[j].Item1] = requests[j].Item2;
                }));
                threads.Peek().Name = ind.ToString();
                threads.Peek().Start();
            }
            while (threads.Count > 0)
            {
                threads.Peek().Join();
                threads.Pop();
            }
        }

        static void Main(string[] args)
        {
            Random rand = new Random();
            foreach (int countRequests in CountsRequests)
            {
                var requests = new (int, int?)[countRequests];
                for (int ind = 0; ind < countRequests; ind++)
                {
                    if (rand.Next(1, 3) == 2)
                        requests[ind] = (rand.Next(1, MaxNumberVertexes), null);
                    else
                        requests[ind] = (rand.Next(1, MaxNumberVertexes), rand.Next(1, MaxNumberVertexes));
                }

                var tree1 = new CoarseSynchronizedBinarySearchTree<int, int>();
                var timer1 = Stopwatch.StartNew();
                ExecRequests<int, int, BinarySearchNode<int, int>, CoarseSynchronizedBinarySearchTree<int, int>>(tree1, requests);
                timer1.Stop();
                Console.WriteLine
                    ($"Building tree with rough synchronization on {countRequests} requests took {timer1.ElapsedMilliseconds} milliseconds.");
                var tree2 = new FineSynchronizedBinarySearchTree<int, int>();
                var timer2 = Stopwatch.StartNew();
                ExecRequests<int, int, LockedBinarySearchNode<int, int>, FineSynchronizedBinarySearchTree<int, int>>(tree2, requests);
                timer2.Stop();
                Console.WriteLine
                    ($"Building tree with fine synchronization on {countRequests} requests took {timer2.ElapsedMilliseconds} milliseconds.\n");
            }
        }
    }
}
