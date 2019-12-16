using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace Task05
{
    public abstract class AbstractTreeTest
    {
        protected abstract ITree<Int32, Int32> CreateTree();

        [Fact]
        public void EmptyTreeTest()
        {
            var tree = CreateTree();

            var value = tree.Find(1);

            Assert.Equal(0, value);
        }

        [Fact]
        public void SimpleFindTest()
        {
            var tree = CreateTree();

            tree.Add(1, 2);
            var value = tree.Find(1);

            Assert.Equal(2, value);
        }

        [Theory]
        [InlineData(1, 1000)]
        [InlineData(10, 1000)]
        public void InsertionTest(Int32 numberOfWorkers, Int32 amountOfWork)
        {
            var tree = CreateTree();
            var distribution = Utils.GetSimpleDistribution(numberOfWorkers, amountOfWork);
            var tasks = Utils.GetInsertionTasks(tree, distribution);

            Utils.RunAll(tasks);

            foreach (var indices in distribution) {
                foreach (var index in indices) {
                    Assert.Equal(index + 1, tree.Find(index));
                }
            }

            //Assert.True(tree.IsValid());
        }
    }
}
