using System;
using System.Collections.Generic;
using NUnit.Framework;

namespace Concurrent_trees.Coarse.Tests
{
    [TestFixture]
    public class SequentialTests
    {
        [Test]
        public void Composite()
        {
            var tree = new Tree();

            for (var i = 0; i < 100; ++i) tree.Insert(i);

            for (var i = 0; i < 100; ++i) Assert.True(tree.Find(i));
            for (var i = 100; i < 200; ++i) Assert.False(tree.Find(i));
        }

        [Test]
        public void Empty()
        {
            var tree = new Tree();

            Assert.False(tree.Find(0));
        }

        [Test]
        public void FalseRoot()
        {
            var tree = new Tree();

            tree.Insert(1);

            Assert.False(tree.Find(0));
        }

        [Test]
        public void Random()
        {
            var tree = new Tree();
            var random = new Random();

            var keys = new List<int>();
            for (var i = 0; i < 100; ++i)
            {
                keys.Add(random.Next(0, 100));
                tree.Insert(keys[i]);
            }

            foreach (var key in keys) Assert.True(tree.Find(key));
        }

        [Test]
        public void Reverse()
        {
            var tree = new Tree();

            for (var i = 0; i < 100; ++i) tree.Insert(i);

            for (var i = 99; i >= 0; --i) Assert.True(tree.Find(i));
        }

        [Test]
        public void RootLeft()
        {
            var tree = new Tree();

            tree.Insert(1);
            tree.Insert(0);

            Assert.True(tree.Find(0));
        }

        [Test]
        public void RootRight()
        {
            var tree = new Tree();

            tree.Insert(-1);
            tree.Insert(0);

            Assert.True(tree.Find(0));
        }

        [Test]
        public void Sequential()
        {
            var tree = new Tree();

            for (var i = 0; i < 100; ++i) tree.Insert(i);

            for (var i = 0; i < 100; ++i) Assert.True(tree.Find(i));
        }

        [Test]
        public void TrueRoot()
        {
            var tree = new Tree();

            tree.Insert(0);

            Assert.True(tree.Find(0));
        }
    }
}