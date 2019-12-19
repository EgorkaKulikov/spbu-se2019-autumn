using NUnit.Framework;

namespace Task05.FineTests
{
    [TestFixture]
    public class InsertionTests
    {
        [TestCase(0, 0)]
        public void Insert_RootValueToEmptyFineTree_UpdateTree(int key, int value)
        {
            var tree = new BSTFineGrained<int>();

            tree.Insert(key, value);

            Assert.AreEqual(value, tree.root.value);
            Assert.AreEqual(null, tree.root.left);
            Assert.AreEqual(null, tree.root.right);
        }

        [TestCase(0, 0)]
        public void Insert_ExistentValueToFineTree_ReturnSameTree(int key, int value)
        {
            var tree = new BSTFineGrained<int>();

            tree.Insert(key, value);
            tree.Insert(key, value);

            Assert.AreEqual(value, tree.root.value);
            Assert.AreEqual(null, tree.root.left);
            Assert.AreEqual(null, tree.root.right);
        }

        [TestCase(5, 5, 3, 3, 1, 1)]
        public void Insert_InOrderLeftLeftToFineTree_UpdateTree(int rootKey, int rootValue, int leftKey,
            int leftValue, int leftOfLeftKey, int leftOfLeftValue)
        {
            var tree = new BSTFineGrained<int>();

            tree.Insert(rootKey, rootValue);
            tree.Insert(leftKey, leftValue);
            tree.Insert(leftOfLeftKey, leftOfLeftValue);

            Assert.AreEqual(rootValue, tree.root.value);
            Assert.IsNull(tree.root.right);
            Assert.IsNotNull(tree.root.left);

            Assert.AreEqual(leftValue, tree.root.left.value);
            Assert.IsNull(tree.root.left.right);
            Assert.IsNotNull(tree.root.left.left);

            Assert.AreEqual(leftOfLeftValue, tree.root.left.left.value);
            Assert.IsNull(tree.root.left.left.left);
            Assert.IsNull(tree.root.left.left.right);
        }

        [TestCase(5, 5, 3, 3, 4, 4)]
        public void Insert_InOrderLeftRightToFineTree_UpdateTree(int rootKey, int rootValue, int leftKey,
            int leftValue, int rightOfLeftKey, int rightOfLeftValue)
        {
            var tree = new BSTFineGrained<int>();

            tree.Insert(rootKey, rootValue);
            tree.Insert(leftKey, leftValue);
            tree.Insert(rightOfLeftKey, rightOfLeftValue);

            Assert.AreEqual(rootValue, tree.root.value);
            Assert.IsNull(tree.root.right);
            Assert.IsNotNull(tree.root.left);

            Assert.AreEqual(leftValue, tree.root.left.value);
            Assert.IsNull(tree.root.left.left);
            Assert.IsNotNull(tree.root.left.right);

            Assert.AreEqual(rightOfLeftValue, tree.root.left.right.value);
            Assert.IsNull(tree.root.left.right.left);
            Assert.IsNull(tree.root.left.right.right);
        }

        [TestCase(5, 5, 7, 7, 6, 6)]
        public void Insert_InOrderRightLeftToFineTree_UpdateTree(int rootKey, int rootValue, int rightKey,
            int rightValue, int leftOfRightKey, int leftOfRightValue)
        {
            var tree = new BSTFineGrained<int>();

            tree.Insert(rootKey, rootValue);
            tree.Insert(rightKey, rightValue);
            tree.Insert(leftOfRightKey, leftOfRightValue);

            Assert.AreEqual(rootValue, tree.root.value);
            Assert.IsNull(tree.root.left);
            Assert.IsNotNull(tree.root.right);

            Assert.AreEqual(rightValue, tree.root.right.value);
            Assert.IsNull(tree.root.right.right);
            Assert.IsNotNull(tree.root.right.left);

            Assert.AreEqual(leftOfRightValue, tree.root.right.left.value);
            Assert.IsNull(tree.root.right.left.left);
            Assert.IsNull(tree.root.right.left.right);
        }

        [TestCase(5, 5, 7, 7, 9, 9)]
        public void Insert_InOrderRightRightToFineTree_UpdateTree(int rootKey, int rootValue, int rightKey,
            int rightValue, int rightOfRightKey, int  rightOfRightValue)
        {
            var tree = new BSTFineGrained<int>();

            tree.Insert(rootKey, rootValue);
            tree.Insert(rightKey, rightValue);
            tree.Insert( rightOfRightKey,  rightOfRightValue);

            Assert.AreEqual(rootValue, tree.root.value);
            Assert.IsNull(tree.root.left);
            Assert.IsNotNull(tree.root.right);

            Assert.AreEqual(rightValue, tree.root.right.value);
            Assert.IsNull(tree.root.right.left);
            Assert.IsNotNull(tree.root.right.right);

            Assert.AreEqual( rightOfRightValue, tree.root.right.right.value);
            Assert.IsNull(tree.root.right.right.left);
            Assert.IsNull(tree.root.right.right.right);
        }
    }
}