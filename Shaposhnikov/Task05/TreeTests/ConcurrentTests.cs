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

        private static (Tree, List<int>) InitRandomTree(Tree tree)
        {
            const int min = 0;
            const int max = 10000;
            var c = 0;
            var list = Enumerable
                .Range(0, 1000)
                .Select(i => Rand.Next(min, max))
                .ToList();

            Parallel.ForEach(list, data =>
            {
                tree.FineInsert(data);
                Interlocked.Increment(ref c);
            });

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
        public void StructureValidate()
        {
            var (tree, _) = InitRandomTree(new Tree());
            
            Traverse(tree.Root, node =>
            {
                if (node.Left != null)
                    Assert.True(node.Left.Value < node.Value);
                if (node.Right != null)
                    Assert.True(node.Right.Value > node.Value);
            });
        }
        
        
        [Test]
        public void InsertCompleteness()
        {
            var (tree, listOfAdded) = InitRandomTree(new Tree());

            foreach (var data in listOfAdded)
            {
                Assert.AreEqual(data, tree.FineFind(data).Value);
            }
        }

        [Test]
        public void RootSafety()
        {
            var (tree, listOfAdded) = InitRandomTree(new Tree());
           
            tree.FineInsert(100); //root

            foreach (var data in listOfAdded.Where(data => data != 100))
            {
                tree.FineRemove(data);
            }

            Assert.AreEqual(100, tree.Root.Value);
            Assert.AreEqual(1, tree.GetDepth());
        }
        
        [Test]
        public void RemovedItemsAbsence()
        {
            var lisеOfRemoved = new List<int>();
            var tree = new Tree();
            
            var mtxRemoved = new Mutex();

            var inputData = Enumerable
                .Range(0, 1000)
                .ToList();

            Parallel.ForEach(inputData, data =>
            {
                if (data % 3 != 0)
                {
                    tree.FineInsert(data);
                }
                else
                {
                    tree.FineInsert(data);
                    mtxRemoved.WaitOne();
                    lisеOfRemoved.Add(data);
                    mtxRemoved.ReleaseMutex();
                }
            });

            Parallel.ForEach(lisеOfRemoved, body: toRemove => tree.FineRemove(toRemove));

            for (var i = 0; i < 1000; i++)
            {
                if (i % 3 != 0)
                    Assert.AreEqual(i, tree.FineFind(i).Value);
                else
                    Assert.AreEqual(null,tree.FineFind(i)?.Value);
            }
        }

        [Test]
        public void ConcurrentBadRemovesAndFind()
        {
            var tree = new Tree();

            var inputData = Enumerable
                .Range(0, 1000)
                .ToList();

            Parallel.ForEach(inputData, data =>
            {
                switch (data % 3)
                {
                    case 0:
                        tree.FineInsert(data);
                        break;
                    case 1:
                        tree.FineRemove(data);
                        break;
                    default:
                        Assert.AreEqual(null, tree.FineFind(data));
                        break;
                }
            });
            
            for (var i = 0; i < 1000; i++)
            {
                switch (i % 3)
                {
                    case 0:
                        Assert.AreEqual(i, tree.FineFind(i).Value);
                        break;
                    default:
                        Assert.AreEqual(null, tree.FineFind(i));
                        break;
                }
            }
        }

        [Test]
        public void InsertCases()
        {
            var tree = new Tree();
            
            Assert.AreEqual(null, tree.FineFind(100));

            List<int> list;
            (tree, list) = InitRandomTree(tree);

            Parallel.ForEach(list, 
                data =>
                {
                    if (data % 2 == 0)
                        Assert.AreEqual(data, tree.FineFind(data).Value);
                    else
                        tree.FineRemove(data);
                });
        }
        //unit tests
        
        [Test]
        public void RemoveRightNull()
        {
            var tree = new Tree();
            tree.FineInsert(10);
            tree.FineInsert(150);
            tree.FineInsert(50);
            
            tree.FineRemove(150);

            Assert.AreEqual(50, tree.Root.Right.Value);
        }
        
        [Test]
        public void RemoveLeftNull()
        {
            var tree = new Tree();
            tree.FineInsert(10);
            tree.FineInsert(150);
            tree.FineInsert(500);
            
            tree.CoarseRemove(150);

            Assert.AreEqual(500, tree.Root.Right.Value);
        }
        
        [Test]
        public void RemoveNoChild()
        {
            var tree = new Tree();
            tree.FineInsert(10);
            tree.FineInsert(150);

            tree.FineRemove(150);

            Assert.AreEqual(null, tree.Root.Right);
        }
        
        [Test]
        public void RemoveWith2Children()
        {
            var tree = new Tree();
            tree.FineInsert(10);
            tree.FineInsert(150);
            tree.FineInsert(500);
            tree.FineInsert(250);
            tree.FineInsert(149);
            
            tree.FineRemove(150);

            Assert.AreEqual(250, tree.Root.Right.Value);
            Assert.AreEqual(149, tree.CoarseFind(250).Left.Value);
            Assert.AreEqual(500, tree.CoarseFind(250).Right.Value);
        }
        
        [Test]
        public void RootCheck()
        {
            var tree = new Tree();
            tree.FineInsert(100); //root
            (tree, _) = InitRandomTree(tree);
            
            tree.FineInsert(99);
            tree.FineInsert(101);
            Assert.AreEqual(100, tree.Root.Value);
        }
        
        [Test]
        public void RemoveRoot()
        {
            var tree = new Tree();
            tree.FineInsert(100);
            tree.FineInsert(150);
            tree.FineInsert(50);
            tree.FineInsert(101);

            tree.FineRemove(100);

            Assert.AreEqual(101, tree.Root.Value);
            Assert.AreEqual(50, tree.Root.Left.Value);
            Assert.AreEqual(150, tree.Root.Right.Value);
            Assert.AreEqual(2, tree.GetDepth());
        }

        [Test]
        public void RootRightRemove()
        {
            var tree = new Tree();
            
            tree.FineInsert(100);
            tree.FineInsert(150);
            tree.FineRemove(100);
            
            Assert.AreEqual(150, tree.Root.Value);
            Assert.AreEqual(null, tree.FineFind(150).Right);
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
        public void EmptyTree()
        {
            var tree = new Tree();
            tree.FineRemove(100);
            Assert.AreEqual(null, tree.FineFind(100));
        }
    }
}