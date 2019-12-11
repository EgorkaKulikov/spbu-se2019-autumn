using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

namespace Task05
{
    class Program
    {
        static int amountThreads = 8;
        static int maxNumberVertexes = 100000;
        static int[] countsRequests = new[] { 10, 30, 100, 275, 800, 2450, 7300, 22000, 65000, 200000 };

        static public void execRequests<K, V, NT, TT>(TT tree, params (K, V?)[] requests)
            where K : IComparable<K>, new()
            where V : struct
            where NT : Node<K, V, NT>
            where TT : Tree<K, V, NT>
        {
            Stack<Thread> threads = new Stack<Thread>();
            for (int i = 0; i < amountThreads; i++)
            {
                int ind = i;
                threads.Push(new Thread(() =>
                {
                    for (int j = ind; j < requests.Length; j += amountThreads)
                        tree[requests[j].Item1] = requests[j].Item2;
                }));
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
            foreach (int countRequests in countsRequests)
            {
                (int, int?)[] requests = new (int, int?)[countRequests];
                for (int ind = 0; ind < countRequests; ind++)
                {
                    if (rand.Next(1, 2) == 2)
                        requests[ind] = (rand.Next(1, maxNumberVertexes), null);
                    else
                        requests[ind] = (rand.Next(1, maxNumberVertexes), rand.Next(1, maxNumberVertexes));
                }
                RoughlySynchronizedBinarySearchTree<int, int> tree1 = new RoughlySynchronizedBinarySearchTree<int, int>();
                Stopwatch timer1 = Stopwatch.StartNew();
                execRequests<int, int, BinarySearchNode<int, int>, RoughlySynchronizedBinarySearchTree<int, int>>(tree1, requests);
                timer1.Stop();
                Console.WriteLine
                    ($"Building tree with rough synchronization on {countRequests} requests took {timer1.ElapsedMilliseconds} milliseconds.");
                FinelySynchronizedBinarySearchTree<int, int> tree2 = new FinelySynchronizedBinarySearchTree<int, int>();
                Stopwatch timer2 = Stopwatch.StartNew();
                execRequests<int, int, BinarySearchNode<int, int>, FinelySynchronizedBinarySearchTree<int, int>>(tree2, requests);
                timer2.Stop();
                Console.WriteLine
                    ($"Building tree with fine synchronization on {countRequests} requests took {timer2.ElapsedMilliseconds} milliseconds.\n");
            }
        }
    }
}
