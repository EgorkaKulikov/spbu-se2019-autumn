using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Task05;

namespace TreeTests
{
    public class Tests
    {
        [Test]
        public void EmptyCoarseTree()
        {
            var tree = new CoarseBST();
            Parallel.For(0, 100, i => Assert.False(tree.Find(i)));
            Assert.True(tree.IsCorrect());
        }

        [Test]
        public void FullCoarseTree()
        {
            var tree = new CoarseBST();
            Parallel.For(0, 100, i =>  tree.Insert(i));
            Parallel.For(0, 100, i => Assert.True(tree.Find(i)));
            Assert.True(tree.IsCorrect());
        }

        [Test]
        public void RandomCoarseTree()
        {
            var tree = new CoarseBST();
            var toInsert = new List<int>();
            var toFind = new List<Tuple<int, bool>>();
            for (var i = 0; i < 100; i++)
            {
                var rnd = new Random();
                var value = rnd.Next(0, 100);

                toInsert.Add(value);
            }

            for (var i = 0; i < 100; i++)
            {
                var rnd = new Random();
                var nxt = rnd.Next(0, 200);
                var value = Tuple.Create<int, bool>(nxt, toInsert.Contains(nxt));
                toFind.Add(value);
            }

            Parallel.ForEach(toInsert, value => tree.Insert(value));
            Parallel.ForEach(toFind, value => Assert.True(tree.Find(value.Item1) == value.Item2));
            Assert.True(tree.IsCorrect());
        }

        [Test]
        public void EmptyFineTree()
        {
            var tree = new CoarseBST();
            Parallel.For(0, 100, i => Assert.False(tree.Find(i)));
            Assert.True(tree.IsCorrect());
        }

        [Test]
        public void FullFineTree()
        {
            var tree = new FineBST();
            Parallel.For(0, 100, i => tree.Insert(i));
            Parallel.For(0, 100, i => Assert.True(tree.Find(i)));
            Assert.True(tree.IsCorrect());
        }

        [Test]
        public void RandomFineTree()
        {
            var tree = new FineBST();
            var toInsert = new List<int>();
            var toFind = new List<Tuple<int, bool>>();
            for (var i = 0; i < 100; i++)
            {
                var rnd = new Random();
                var value = rnd.Next(0, 100);

                toInsert.Add(value);
            }

            for (var i = 0; i < 100; i++)
            {
                var rnd = new Random();
                var nxt = rnd.Next(0, 200);
                var value = Tuple.Create<int, bool>(nxt, toInsert.Contains(nxt));
                toFind.Add(value);
            }

            Parallel.ForEach(toInsert, value => tree.Insert(value));
            Parallel.ForEach(toFind, value => Assert.True(tree.Find(value.Item1) == value.Item2));
            Assert.True(tree.IsCorrect());
        }
    }
}