using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace Task05
{
    public abstract class ITreeTest
    {
        protected abstract ITree<Int32, Int32> CreateTree();

        protected static Int32[] GetRange(Int32 size, Func<Int32, Int32> func) => Enumerable.Range(0, size).Select(i => func(i)).ToArray();
        protected static Int32[] GetRightBambooRange(Int32 size) => GetRange(size, i => i);
        protected static Int32[] GetLeftBambooRange(Int32 size) => GetRange(size, i => -i);
        protected static Int32[] GetBalancedRange(Int32 size) => GetRange(size, i =>
        {
            Int32 power = (Int32)Math.Log2(i + 1) + 1;
            Int32 denumerator = (Int32)Math.Pow(2, power);
            Int32 first = denumerator / 2 - 1;

            return size * (1 + (i - first) * 2) / denumerator;
        });

        protected static Int32[][] ranges = new Int32[][] {
            GetRightBambooRange(0),
            GetRightBambooRange(1024),
            GetLeftBambooRange(1024),
            GetBalancedRange(1024),
        };

        [Fact]
        public void EmptyTreeTest()
        {
            var tree = CreateTree();

            var value = tree.Find(1);

            Assert.Equal(0, value);
        }

        [Theory]
        [InlineData(0, 1)]
        [InlineData(1, 1)]
        [InlineData(2, 1)]
        [InlineData(3, 1)]
        [InlineData(0, 8)]
        [InlineData(1, 8)]
        [InlineData(2, 8)]
        [InlineData(3, 8)]
        public void InsertionTest(Int32 rangeIndex, Int32 numberOfTasks)
        {
            var tree = CreateTree();
            var range = ranges[rangeIndex];

            var tasks = new Task[numberOfTasks];
            var chunkSize = range.Length / numberOfTasks;

            for (Int32 i = 0; i < numberOfTasks; i++)
            {
                var num = i;
                tasks[i] = Task.Run(() =>
                {
                    foreach (var index in Enumerable.Range(num * chunkSize, chunkSize))
                    {
                        tree.Add(range[index], range[index] + 1);
                    }
                });
            }

            Task.WaitAll(tasks);

            foreach (var index in range)
            {
                Assert.Equal(index + 1, tree.Find(index));
            }
        }

        [Theory]
        [InlineData(0, 1)]
        [InlineData(1, 1)]
        [InlineData(2, 1)]
        [InlineData(3, 1)]
        [InlineData(0, 10)]
        [InlineData(1, 10)]
        [InlineData(2, 10)]
        [InlineData(3, 10)]
        public void DeletionTest(Int32 rangeIndex, Int32 numberOfTasks)
        {
            var tree = CreateTree();
            var range = ranges[rangeIndex];

            foreach (var index in range)
            {
                tree.Add(index, index + 1);
            }

            var tasks = new Task[numberOfTasks];
            var chunkSize = range.Length / numberOfTasks;

            for (Int32 i = 0; i < numberOfTasks; i++)
            {
                var num = i;
                tasks[i] = Task.Run(() =>
                {
                    foreach (var index in Enumerable.Range(num * chunkSize, chunkSize))
                    {
                        if (index % 2 == 0)
                        {
                            tree.Delete(range[index]);
                        }
                    }
                });
            }

            Task.WaitAll(tasks);

            for (Int32 i = 0; i < range.Length; i++)
            {
                if (i % 2 == 1)
                {
                    Assert.Equal(range[i] + 1, tree.Find(range[i]));
                } else {
                    Assert.Equal(0, tree.Find(range[i]));
                }
            }
        }
    }
}
