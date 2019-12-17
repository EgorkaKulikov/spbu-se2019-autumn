using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using NUnit.Framework;
using Task05;

namespace TestProject1
{
    public class Tests
    {
    
        private readonly List<int> _addList = new List<int>();
        private readonly List<int> _findList = new List<int>();
        
        [SetUp]
        public void Setup()
        {
            for (var i = 0; i < 100; i++)
            {
                var random= new Random();
                var value = random.Next(0, 1000);
                _addList.Add(value);
                _findList.Add(value);
            }

            for (var i = 100; i < 200; i++)
            {
                var random= new Random();
                var value = random.Next(1001, 2000);
                _findList.Add(value);  
            }
        }

        [Test]
        public void TestOptimisticGrainedTree()
        {
            var tree = new OptimisticGrainedBinaryTree();
            var tasks = new List<Task>();
            
            foreach (var value in _addList)
            {
                tasks.Add(Task.Run(() =>tree.Insert(value)));
            }
            
            foreach (var value in _findList)
            {
                tasks.Add(Task.Run(() =>tree.Find(value)));
            }
            
            Task.WaitAll(tasks.ToArray());

            foreach (var value in _addList)
            {
                Assert.True(tree.Find(value));
            }
            
            Assert.True(CheckStructure(tree.Root));
        }
        
        [Test]
        public void TestCoarseGrainedTree()
        {
            var tree = new CoarseGrainedBinaryTree();
            var tasks = new List<Task>();
            
            foreach (var value in _addList)
            {
                tasks.Add(Task.Run(() =>tree.Insert(value)));
            }
            
            foreach (var value in _findList)
            {
                tasks.Add(Task.Run(() =>tree.Find(value)));
            }
            
            Task.WaitAll(tasks.ToArray());

            foreach (var value in _addList)
            {
                Assert.True(tree.Find(value));
            }
            
            Assert.True(CheckStructure(tree.Root));
        }
        
        private bool CheckStructure(BinaryTreeNode node)
        {
            if (node == null)
            {
                return true;
            }

            var checkLeft = CheckStructure(node.Left);
            var checkRight = CheckStructure(node.Right);
            
            if (node.Left == null && node.Right== null)
            {
                return true;
            }

            if (node.Left == null || node.Right == null)
            {
                if (node.Left == null)
                {
                    return (node.Key < node.Right?.Key) && checkRight;
                }

                return (node.Key > node.Left?.Key) && checkLeft;
            }

            return (node.Left?.Key < node.Key && node.Key < node.Right?.Key) &&
                   checkLeft && checkRight;
        }
    }
}