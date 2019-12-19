using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;

namespace Task05.FineTests
{
    [TestFixture]
    public class ParallelScenarios
    {
        private readonly Random _random = new Random();
        
        [TestCase(30, 1000, 1000, 10)]
        public void InsertSearchConcurrently_ManyValuesToFineTree_UpdateTree(int workers, int maxSize, 
            int maxValue, int maxTimeout)
        {
            var expectedTree = new BSTFineGrained<int>();
            var actualTree = new BSTFineGrained<int>();
            var elementsToInsert = new ConcurrentQueue<(int,int)>();
            for (int i = 0; i < maxSize; ++i)
            {
                var value = _random.Next(maxValue);
                elementsToInsert.Enqueue((value,value));
                expectedTree.Insert(value,value);
            }
            var elementsToSearch = new ConcurrentQueue<(int,int)>(elementsToInsert);
            Task[] tasks = new Task[workers];   
            
            for (int i = 0; i < workers; ++i)
            {
                if (_random.Next(2) == 0)
                {
                    // Create worker to test insertion
                    tasks[i] = Task.Run(() =>
                    {
                        while (!elementsToInsert.IsEmpty)
                        {
                            if (elementsToInsert.TryDequeue(out (int,int) pair))
                            {
                                actualTree.Insert(pair.Item1, pair.Item2);
                            }
                            Thread.Sleep(_random.Next(maxTimeout));
                        }
                    });
                }
                else
                {
                    // Create worker to test search
                    tasks[i] = Task.Run(() =>
                    {
                        while (!elementsToSearch.IsEmpty)
                        {
                            if (elementsToSearch.TryDequeue(out (int,int) pair))
                            {
                                actualTree.Search(pair.Item1);
                            }
                            Thread.Sleep(_random.Next(maxTimeout));
                        }
                    });
                }
            }
            
            Task.WaitAll(tasks);
            Assert.IsTrue(Utils.ContentEquals(actualTree, expectedTree));
            Assert.IsTrue(Utils.ValidateStructure(actualTree));
        }
        
        [TestCase(30, 1000, 1000, 10)]
        public void InsertSequentially_DeleteSearchConcurrently_ManyValuesToFineTree_UpdateTree(int workers, 
            int maxSize, int maxValue, int maxTimeout)
        {
            var expectedTree = new BSTFineGrained<int>();
            var actualTree = new BSTFineGrained<int>();
            var elementsToInsert = new ConcurrentQueue<(int,int)>();
            var elementsToDelete = new ConcurrentQueue<(int,int)>();
            for (int i = 0; i < maxSize; ++i)
            {
                var value = _random.Next(maxValue);
                elementsToInsert.Enqueue((value,value));
                actualTree.Insert(value,value);
                expectedTree.Insert(value,value);
                value = _random.Next(maxValue);
                elementsToDelete.Enqueue((value,value));
            }
            var elementsToSearch = new ConcurrentQueue<(int,int)>(elementsToInsert);
            foreach ((int,int) pair in elementsToDelete)
            {
                expectedTree.Delete(pair.Item1);
            }
            Task[] tasks = new Task[workers];
            
            for (int i = 0; i < workers; ++i)
            {
                if (_random.Next(2) == 0)
                {
                    // Create worker to test removing
                    tasks[i] = Task.Run(() =>
                    {
                        while (!elementsToDelete.IsEmpty)
                        {
                            if (elementsToDelete.TryDequeue(out (int,int) pair))
                            {
                                actualTree.Delete(pair.Item1);
                            }
                            Thread.Sleep(_random.Next(maxTimeout));
                        }
                    });
                }
                else
                {
                    // Create worker to test search
                    tasks[i] = Task.Run(() =>
                    {
                        while (!elementsToSearch.IsEmpty)
                        {
                            if (elementsToSearch.TryDequeue(out (int,int) pair))
                            {
                                actualTree.Search(pair.Item1);
                            }
                            Thread.Sleep(_random.Next(maxTimeout));
                        }
                    });
                }
            }
            
            Task.WaitAll(tasks);
            Assert.IsTrue(Utils.ContentEquals(actualTree, expectedTree));
            Assert.IsTrue(Utils.ValidateStructure(actualTree));
        }
        
        [TestCase(100, 10000, 10000, 10)]
        public void InsertDeleteSearchConcurrently_ManyValuesToFineTree_UpdateTree(int workers, 
            int maxSize, int maxValue, int maxTimeout)
        {
            var actualTree = new BSTFineGrained<int>();
            var elementsToInsert = new ConcurrentQueue<(int,int)>();
            var elementsToDelete = new ConcurrentQueue<(int,int)>();
            for (int i = 0; i < maxSize; ++i)
            {
                var value = _random.Next(maxValue);
                elementsToInsert.Enqueue((value,value));
                value = _random.Next(maxValue);
                elementsToDelete.Enqueue((value,value));
            }
            var elementsToSearch = new ConcurrentQueue<(int,int)>(elementsToInsert);
            Task[] tasks = new Task[workers];
            
            for (int i = 0; i < workers; ++i)
            {
                int actionId = _random.Next(3);
                if (actionId == 0)
                {
                    // Create worker to test insertion
                    tasks[i] = Task.Run(() =>
                    {
                        while (!elementsToInsert.IsEmpty)
                        {
                            if (elementsToInsert.TryDequeue(out (int,int) pair))
                            {
                                actualTree.Insert(pair.Item1, pair.Item2);
                            }
                            Thread.Sleep(_random.Next(maxTimeout));
                        }
                    });
                }
                else if (actionId == 1)
                {
                    // Create worker to test removing
                    tasks[i] = Task.Run(() =>
                    {
                        while (!elementsToDelete.IsEmpty)
                        {
                            if (elementsToDelete.TryDequeue(out (int,int) pair))
                            {
                                actualTree.Delete(pair.Item1);
                            }
                            Thread.Sleep(_random.Next(maxTimeout));
                        }
                    });
                }
                else 
                {
                    // Create worker to test search
                    tasks[i] = Task.Run(() =>
                    {
                        while (!elementsToSearch.IsEmpty)
                        {
                            if (elementsToSearch.TryDequeue(out (int,int) pair))
                            {
                                actualTree.Search(pair.Item1);
                            }
                            Thread.Sleep(_random.Next(maxTimeout));
                        }
                    });
                }
            }
            
            Task.WaitAll(tasks);
            Assert.IsTrue(Utils.ValidateStructure(actualTree));
        }
        
        [TestCase(30, 1000, 10000, 10)]
        public void InsertSequentially_InsertSearchConcurrently_ManyValuesToFineTree_UpdateTree(int workers, 
            int maxSize, int maxValue, int maxTimeout)
        {
            var actualTree = new BSTFineGrained<int>();
            var elementsToInsertSeq = new ConcurrentQueue<(int,int)>();
            var elementsToInsertConcur = new ConcurrentQueue<(int,int)>();
            for (int i = 0; i < maxSize; ++i)
            {
                var value = _random.Next(maxValue);
                elementsToInsertSeq.Enqueue((value,value));
                actualTree.Insert(value, value);
                value = _random.Next(maxValue);
                elementsToInsertConcur.Enqueue((value,value));
            }
            var elementsToSearch = new ConcurrentQueue<(int,int)>(elementsToInsertSeq);
            Task[] tasks = new Task[workers];
            
            for (int i = 0; i < workers; ++i)
            {
                if (_random.Next(2) == 0)
                {
                    // Create worker to test insertion
                    tasks[i] = Task.Run(() =>
                    {
                        while (!elementsToInsertConcur.IsEmpty)
                        {
                            if (elementsToInsertConcur.TryDequeue(out (int,int) pair))
                            {
                                actualTree.Insert(pair.Item1, pair.Item2);
                            }
                            Thread.Sleep(_random.Next(maxTimeout));
                        }
                    });
                }
                else
                {
                    // Create worker to test search
                    tasks[i] = Task.Run(() =>
                    {
                        while (!elementsToSearch.IsEmpty)
                        {
                            if (elementsToSearch.TryDequeue(out (int,int) pair))
                            {
                                Assert.AreEqual(pair.Item2, actualTree.Search(pair.Item1));
                            }
                            Thread.Sleep(_random.Next(maxTimeout));
                        }
                    });
                }
            }
            
            Task.WaitAll(tasks);
            Assert.IsTrue(Utils.ValidateStructure(actualTree));
        }
    }
}