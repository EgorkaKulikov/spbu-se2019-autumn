using System;
using System.Linq;
using NUnit.Framework;
using Task05;

namespace TreeTests
{
    [TestFixture]
    public class SeqTests
    {
        private static readonly Random Rand = new Random();

        private static Tree InitTree(Tree tree)
        {
            const int min = 0;
            const int max = 10000;
            var inputData = Enumerable
                .Repeat(0, 2000)
                .Select(i => Rand.Next(min, max))
                .ToList();
            foreach (var data in inputData)
            {
                tree.CoarseInsert(data);
            }

            return tree;
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
        
        //There go tests for BST functions and properties' correctness
        [Test]
        public void RootCheck()
        {
            var tree = new Tree();
            tree.CoarseInsert(100); //root
            tree = InitTree(tree);
            
            tree.CoarseInsert(99);
            tree.CoarseInsert(101);
            tree.CoarseRemove(99);
            tree.CoarseRemove(101);
            Assert.AreEqual(100, tree.Root.Value);
        }

        [Test]
        public void InsertCompleteness()
        {
            var tree = new Tree();
            var inputData = Enumerable
                .Range(0, 200)
                .OrderBy(a => Guid.NewGuid())
                .ToList();
            inputData.ForEach(data => tree.CoarseInsert(data));
            for (var i = 0; i < 200; i++)
            {
                Assert.AreEqual(i, tree.CoarseFind(i)?.Value);
            }
        }
        
        [Test]
        public void RemovedItemsAbsence()
        {
            var tree = new Tree();
            var inputData = Enumerable
                .Range(0, 200)
                .OrderBy(a => Guid.NewGuid())
                .ToList();
            inputData.ForEach(data => tree.CoarseInsert(data));
            var toRemoveData = Enumerable.Range(20, 80).ToList();
            toRemoveData.ForEach(data => tree.CoarseRemove(data));
            for (var i = 20; i < 80; i++)
            {
                Assert.False(tree.CoarseFind(i)?.Value == i);
            }
        }

        [Test]
        public void RelationAssert()
        {
            var tree = InitTree(new Tree());
            
            Traverse(tree.Root, node =>
            {
                if (node.Left != null)
                    Assert.True(node.Left.Value < node.Value);
                if (node.Right != null)
                    Assert.True(node.Right.Value > node.Value);
            });
        }
        
        [Test]
        public void BadCalls()
        {
            var tree = new Tree();

            tree.CoarseRemove(100);
            tree.CoarseInsert(10);
            tree.CoarseRemove(50);
            Assert.AreEqual(10, tree.CoarseFind(10)?.Value);
            Assert.AreEqual(null, tree.CoarseFind(100));
        }

        //Here go unit tests
        [Test]
        public void RemoveRightNull()
        {
            var tree = new Tree();
            tree.CoarseInsert(10);
            tree.CoarseInsert(150);
            tree.CoarseInsert(50);
            
            tree.CoarseRemove(150);

            Assert.AreEqual(50, tree.Root.Right.Value);
        }
        
        [Test]
        public void RemoveLeftNull()
        {
            var tree = new Tree();
            tree.CoarseInsert(10);
            tree.CoarseInsert(150);
            tree.CoarseInsert(500);
            
            tree.CoarseRemove(150);

            Assert.AreEqual(500, tree.Root.Right.Value);
        }
        
        [Test]
        public void RemoveNoChild()
        {
            var tree = new Tree();
            tree.CoarseInsert(10);
            tree.CoarseInsert(150);

            tree.CoarseRemove(150);

            Assert.AreEqual(null, tree.Root.Right);
        }
        
        [Test]
        public void RemoveWith2Children()
        {
            var tree = new Tree();
            tree.CoarseInsert(10);
            tree.CoarseInsert(150);
            tree.CoarseInsert(500);
            tree.CoarseInsert(250);
            tree.CoarseInsert(149);
            
            tree.CoarseRemove(150);

            Assert.AreEqual(250, tree.Root.Right.Value);
            Assert.AreEqual(149, tree.CoarseFind(250).Left.Value);
            Assert.AreEqual(500, tree.CoarseFind(250).Right.Value);
        }
        
        [Test]
        public void RemoveComplex()
        {
            var tree = new Tree();
            tree.CoarseInsert(10);
            tree.CoarseInsert(150);
            tree.CoarseInsert(500);
            tree.CoarseInsert(250);
            tree.CoarseInsert(255);
            tree.CoarseInsert(254);
            tree.CoarseInsert(149);
            
            tree.CoarseRemove(150);

            Assert.AreEqual(250, tree.Root.Right.Value);
            Assert.AreEqual(149, tree.CoarseFind(250).Left.Value);
            Assert.AreEqual(500, tree.CoarseFind(250).Right.Value);
            Assert.AreEqual(255, tree.CoarseFind(500).Left.Value);
            Assert.AreEqual(254, tree.CoarseFind(255).Left.Value);
        }
        
        [Test]
        public void RemoveRoot()
        {
            var tree = new Tree();
            tree.CoarseInsert(100);
            tree.CoarseInsert(150);
            tree.CoarseInsert(50);
            tree.CoarseInsert(101);

            tree.CoarseRemove(100);

            Assert.AreEqual(101, tree.Root.Value);
            Assert.AreEqual(50, tree.Root.Left.Value);
            Assert.AreEqual(150, tree.Root.Right.Value);
            Assert.AreEqual(2, tree.GetDepth());
        }
    }
}