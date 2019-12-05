using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using NUnit.Framework;
using Task05;

namespace TestProject1
{
    public class Tests
    {
        private List<int> toAdd = new List<int>();
        private List<int> toDelete = new List<int>();
        private List<int> toFind = new List<int>();
        
        [SetUp]
        public void Setup()
        {
            for (var i = 0; i < 100; i++)
            {
                var rng = new Random();
                toAdd.Add(rng.Next(0, 100));
            }
            
            for (var i = 0; i < 50; i++)
            {
                var rng = new Random();
                toDelete.Add(rng.Next(0, 100));
            }
            
            for (var i = 0; i < 100; i++)
            {
                var rng = new Random();
                toFind.Add(rng.Next(0, 200));
            }
        }

        [Test]
        public void TestCoarse()
        {
            var tree = new CoarseGrainedBST();
            var tasks = new List<Task>();
            foreach (var value in toAdd)
            {
                tasks.Add(Task.Run(() => tree.add(value)));
            }
            
            foreach (var value in toDelete)
            {
                tasks.Add(Task.Run(() => tree.delete(value)));
            }
            
            foreach (var value in toFind)
            {
                tasks.Add(Task.Run(() => tree.find(value)));
            }

            Task.WaitAll(tasks.ToArray());
            Assert.True(tree.verify());
        }
        
        [Test]
        public void TestFine()
        {
            var tree = new FineGrainedBST();
            var tasks = new List<Task>();
            foreach (var value in toAdd)
            {
                tasks.Add(Task.Run(() => tree.add(value)));
            }
            
            foreach (var value in toDelete)
            {
                tasks.Add(Task.Run(() => tree.delete(value)));
            }
            
            foreach (var value in toFind)
            {
                tasks.Add(Task.Run(() => tree.find(value)));
            }

            Task.WaitAll(tasks.ToArray());
            Assert.True(tree.verify());
        }
    }
}