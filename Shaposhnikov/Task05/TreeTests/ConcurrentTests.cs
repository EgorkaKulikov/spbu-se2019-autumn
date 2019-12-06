using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using Task05;

namespace TreeTests
{
    [TestFixture]
    public class ConcurrentTests
    {
        private static readonly Random Rand = new Random();

        private static (Tree, List<int>) InitTree(Tree tree)
        {
            const int min = 0;
            const int max = 10000;
            var c = 0;
            var list = Enumerable
                .Range(0, 1000)
                .Select(i => Rand.Next(min, max))
                .ToList();

            var res = Parallel.ForEach(list, data =>
            {
                tree.FineInsert(data);
                Interlocked.Increment(ref c);
            });
            while (!res.IsCompleted)
            {
            }

            Assert.AreEqual(1000, c);
            
            return (tree, list);
        }

        private static void Traverse(Node parent, Action<Node> a)
        {
            while (true)
            {
                if (parent == null) return;
                Traverse(parent.Left, a);
                a(parent);
                parent = parent.Right;
            }
        }
        
        [Test]
        public void RootLastDeletion()
        {
            var tree = new Tree();
            tree.FineInsert(100);
            tree.FineInsert(101);
            tree.FineInsert(99);

            tree.FineRemove(101);
            tree.FineRemove(100);
            Assert.AreEqual(99, tree.Root.Value);
        }
        
        [Test]
        public void InsertCompleteness()
        {
            var (tree, listOfAdded) = InitTree(new Tree());

            foreach (var data in listOfAdded)
            {
                Assert.AreEqual(data, tree.FineFind(data).Value);
            }
        }
        
        [Test]
        public void RootCheck()
        {
            var tree = new Tree();
            tree.FineInsert(100); //root
            (tree, _) = InitTree(tree);
            
            tree.FineInsert(99);
            tree.FineInsert(101);
            Assert.AreEqual(100, tree.Root.Value);
        }

        [Test]
        public void RootSafety()
        {
            var (tree, listOfAdded) = InitTree(new Tree());
           
            tree.FineInsert(100); //root

            foreach (var data in listOfAdded.Where(data => data != 100))
            {
                tree.FineRemove(data);
            }

            Thread.Sleep(1000);
            Assert.AreEqual(100, tree.Root.Value);
            Assert.AreEqual(1, tree.GetDepth());
        }
        
        /*[Test]
        public void RemovedItemsAbsence()
        {
            var lisOfRemoved = new List<int>();
            var listOfAdded = new List<int>();
            var tree = new Tree();

            const int min = 0;
            const int max = 10000;
            var inputData = Enumerable
                .Range(0, 1000)
                .ToList();

            Parallel.ForEach(inputData, data =>
            {
                if (data % 3 != 0)
                {
                    listOfAdded.Add(data);
                    tree.FineInsert(data);
                }
                else
                {
                    lisOfRemoved.Add(data);
                    tree.FineInsert(data);
                }
            });

            Parallel.ForEach(lisOfRemoved, body: toRemove => tree.FineRemove(toRemove));

            for (var i = 0; i < 1000; i++)
            {
                if (i % 3 != 0)
                    Assert.AreEqual(i, tree.FineFind(i).Value);
                else
                    Assert.AreEqual(null,tree.FineFind(i)?.Value);
            }
        }*/
    }
}