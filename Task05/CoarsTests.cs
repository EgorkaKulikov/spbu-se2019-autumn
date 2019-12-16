using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using NUnit.Framework;

namespace Task05
{
    public class CoarsTests
    {
        [Test]
        public void MutexTest()
        {
            List<Tuple<int, int>> insertList = new List<Tuple<int, int>>();
            List<int> findList = new List<int>();
            var tree = new CoarseBinaryTree();
            var tasks = new List<Task>();

            for (int i = 0; i < 100; i++)
            {
                var random = new Random();
                insertList.Add(new Tuple<int, int>(random.Next(0, 100),random.Next(0, 100)));
                findList.Add(random.Next(0, 100));
            }

            foreach (var node in insertList)
                tasks.Add(Task.Run(() => tree.insert(node.Item1, node.Item2)));

            foreach (var key in findList)
                tasks.Add(Task.Run(() => tree.find(key)));

            Task.WaitAll(tasks.ToArray());
            Assert.True(tree.mutex.WaitOne());
        }
        
        [Test]
        public void IsBinaryTreeTest()
        {
            List<Tuple<int, int>> insertList = new List<Tuple<int, int>>();
            List<int> findList = new List<int>();
            var tree = new CoarseBinaryTree();
            var tasks = new List<Task>();

            for (int i = 0; i < 100; i++)
            {
                var random = new Random();
                insertList.Add(new Tuple<int, int>(random.Next(0, 100),random.Next(0, 100)));
                findList.Add(random.Next(0, 100));
            }

            foreach (var node in insertList)
                tasks.Add(Task.Run(() => tree.insert(node.Item1, node.Item2)));

            foreach (var key in findList)
                tasks.Add(Task.Run(() => tree.find(key)));

            Task.WaitAll(tasks.ToArray());
            Assert.True(tree.isBinaryTree(tree.root));
        }
    }
}