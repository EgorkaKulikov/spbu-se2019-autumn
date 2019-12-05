using System;
using NUnit.Framework;

namespace Task05
{
    [TestFixture]
    public class Tests
    {
        [Test]
        public void InsertionSearchingTestFG()
        {
            var helper = new TestHelper<FineGrainedTree<int, int>>();
            Assert.True(helper.FoundAllInserted(TestConfig.TestNumKeys));
        }
        
        [Test]
        public void InsertionSearchingTestCG()
        {
            var helper = new TestHelper<CoarseGrainedTree<int, int>>();
            Assert.True(helper.FoundAllInserted(TestConfig.TestNumKeys));
        }
        
        [Test]
        public void ConcInsertionRemovalTestCG()
        {
            var helper = new TestHelper<CoarseGrainedTree<int, int>>();
            Assert.True(helper.IsEmptyAfterConcurrentInsertionRemoval(TestConfig.TestNumKeys
                , TestConfig.TestNumTasks));
        }
        
        [Test]
        public void InsertionRemovalTestFG()
        {
            var helper = new TestHelper<FineGrainedTree<int, int>>();
            Assert.True(helper.IsEmptyAfterInsertionRemoval(TestConfig.TestNumKeys));
        }
        
        [Test]
        public void InsertionRemovalTestCG()
        {
            var helper = new TestHelper<CoarseGrainedTree<int, int>>();
            Assert.True(helper.IsEmptyAfterInsertionRemoval(TestConfig.TestNumKeys));
        }
    }
}