using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using Parallel_Trees;

namespace Tests_Trees
{
    internal class TestsParallelBinaryTree
    {
        private static readonly Random Random = new Random();
        
        internal static void InsertFind_EmptyTree(BinaryTree<int, int> tree)
        {
            //initialization
            var expectedTree = tree;
            var actualTree = tree;
            var amountWorkers = 30;
            var amountNodes = 1000;
            var tasks = new Task[amountWorkers];
            var elementsToIns = new ConcurrentQueue<int>();
            var elementsToFind = new ConcurrentQueue<int>();
            var upperBoundValue = 10_000;
            var maxTimeout = 1000;

            for (int index = 0; index < amountNodes; ++index)
            {
                var value = Random.Next(upperBoundValue);
                elementsToIns.Enqueue(value);
                elementsToFind.Enqueue(value);
                expectedTree.Insert(value, value);
            }

            //action
            for (int i = 0; i < amountWorkers; ++i)
            {
                if (Random.Next(2) == 0)
                {
                    tasks[i] = Task.Run(() =>
                    {
                        while (!elementsToIns.IsEmpty)
                        {
                            if (elementsToIns.TryDequeue(out var value))
                            {
                                actualTree.Insert(value, value);
                            }

                            Thread.Sleep(Random.Next(maxTimeout));
                        }
                    });
                }
                else
                {
                    tasks[i] = Task.Run(() =>
                    {
                        while (!elementsToFind.IsEmpty)
                        {
                            if (elementsToFind.TryDequeue(out var value))
                            {
                                actualTree.Find(value);
                            }

                            Thread.Sleep(Random.Next(maxTimeout));
                        }
                    });
                }
            }

            Task.WaitAll(tasks);

            //assert
            Assert.IsTrue(TestUtils.ContentEquals(actualTree, expectedTree));
            Assert.IsTrue(TestUtils.CheckRule(actualTree));
        }

        internal static void FindRemove_FilledTree(BinaryTree<int, int> tree)
        {
            //initialization
            var expectedTree = tree;
            var actualTree = tree;
            var amountWorkers = 30;
            var amountNodes = 1000;
            var tasks = new Task[amountWorkers];
            var elementsToRemove = new ConcurrentQueue<int>();
            var elementsToFind = new ConcurrentQueue<int>();
            var upperBoundValue = 10_000;
            var maxTimeout = 1000;

            for (int index = 0; index < amountNodes; ++index)
            {
                var value = Random.Next(upperBoundValue);
                elementsToFind.Enqueue(value);
                expectedTree.Insert(value, value);
                actualTree.Insert(value, value);
                if (Random.Next(2) == 0) elementsToRemove.Enqueue(value);
            }

            foreach (var elem in elementsToRemove)
            {
                expectedTree.Remove(elem);
            }

            //action
            for (int i = 0; i < amountWorkers; ++i)
            {
                if (Random.Next(2) == 0)
                {
                    tasks[i] = Task.Run(() =>
                    {
                        while (!elementsToFind.IsEmpty)
                        {
                            if (elementsToFind.TryDequeue(out var value))
                            {
                                actualTree.Find(value);
                            }

                            Thread.Sleep(Random.Next(maxTimeout));
                        }
                    });
                }
                else
                {
                    tasks[i] = Task.Run(() =>
                    {
                        while (!elementsToRemove.IsEmpty)
                        {
                            if (elementsToRemove.TryDequeue(out var value))
                            {
                                actualTree.Remove(value);
                            }

                            Thread.Sleep(Random.Next(maxTimeout));
                        }
                    });
                }
            }

            Task.WaitAll(tasks);

            //assert
            Assert.IsTrue(TestUtils.ContentEquals(actualTree, expectedTree));
            Assert.IsTrue(TestUtils.CheckRule(actualTree));
        }

        internal static void InsertFindRemove(BinaryTree<int, int> tree)
        {
            //initialization
            var amountWorkers = 30;
            var amountNodes = 1000;
            var tasks = new Task[amountWorkers];
            var elementsToIns = new ConcurrentQueue<int>();
            var elementsToRemove = new ConcurrentQueue<int>();
            var elementsToFind = new ConcurrentQueue<int>();
            var upperBoundValue = 10_000;
            var maxTimeout = 1000;

            for (int index = 0; index < amountNodes; ++index)
            {
                var value = Random.Next(upperBoundValue);
                elementsToIns.Enqueue(value);
                elementsToFind.Enqueue(value);
                if (Random.Next(2) == 0) elementsToRemove.Enqueue(value);
            }
            
            //action
            for (int i = 0; i < amountWorkers; ++i)
            {
                switch (Random.Next(3))
                {
                    case 0:
                    {
                        tasks[i] = Task.Run(() =>
                        {
                            while (!elementsToIns.IsEmpty)
                            {
                                if (elementsToIns.TryDequeue(out var value))
                                {
                                    tree.Insert(value, value);
                                }

                                Thread.Sleep(Random.Next(maxTimeout));
                            }
                        });
                        break;
                    }
                    case 1:
                    {
                        tasks[i] = Task.Run(() =>
                        {
                            while (!elementsToFind.IsEmpty)
                            {
                                if (elementsToFind.TryDequeue(out var value))
                                {
                                    tree.Find(value);
                                }

                                Thread.Sleep(Random.Next(maxTimeout));
                            }
                        });
                        break;
                    }
                    case 2:
                    {
                        tasks[i] = Task.Run(() =>
                        {
                            while (!elementsToRemove.IsEmpty)
                            {
                                if (elementsToRemove.TryDequeue(out var value))
                                {
                                    tree.Remove(value);
                                }

                                Thread.Sleep(Random.Next(maxTimeout));
                            }
                        });
                        break;
                    }
                }
            }

            Task.WaitAll(tasks);

            //assert
            Assert.IsTrue(TestUtils.CheckRule(tree));
        }
    }
}