using NUnit.Framework;

namespace Task05.CoarseTests
{
    [TestFixture]
    public class DeleteTests
    {
        [TestCase(0, 0)]
        public void Delete_RootInCoarseTree_ReturnEmptyTree(int key, int value)
        {
            var tree = new BSTCoarseGrained<int>();
            tree.Insert(key, value);

            tree.Delete(key);

            Assert.IsNull(tree.root);
        }

        [TestCase(0, 0)]
        public void Delete_NonExistentNodeInCoarseTree_ReturnEmptyTree(int key, int value)
        {
            var tree = new BSTCoarseGrained<int>();
            tree.Insert(key, value);

            Assert.DoesNotThrow(() => tree.Delete(value + 1));

            Assert.AreEqual(value, tree.root.value);
            Assert.IsNull(tree.root.right);
            Assert.IsNull(tree.root.left);
        }

        [TestCase(5, 5, 3, 3, 1, 1)]
        public void Delete_RootInCoarseTree_SetNewRootFromLeft(int rootKey, int rootValue, int leftKey,
            int leftValue, int leftOfLeftKey, int leftOfLeftValue)
        {
            var tree = new BSTCoarseGrained<int>();
            tree.Insert(rootKey, rootValue);
            tree.Insert(leftKey, leftValue);
            tree.Insert(leftOfLeftKey, leftOfLeftValue);

            tree.Delete(rootKey);

            Assert.AreEqual(leftValue, tree.root.value);
            Assert.IsNull(tree.root.right);
            Assert.IsNotNull(tree.root.left);

            Assert.AreEqual(leftOfLeftValue, tree.root.left.value);
            Assert.IsNull(tree.root.left.right);
            Assert.IsNull(tree.root.left.left);
        }

        [TestCase(5, 5, 7, 7, 9, 9)]
        public void Delete_RootInCoarseTree_SetNewRootFromRight(int rootKey, int rootValue, int rightKey,
            int rightValue, int rightOfRightKey, int rightOfRightValue)
        {
            var tree = new BSTCoarseGrained<int>();
            tree.Insert(rootKey, rootValue);
            tree.Insert(rightKey, rightValue);
            tree.Insert(rightOfRightKey, rightOfRightValue);

            tree.Delete(rootKey);

            Assert.AreEqual(rightValue, tree.root.value);
            Assert.IsNull(tree.root.left);
            Assert.IsNotNull(tree.root.right);

            Assert.AreEqual(rightOfRightValue, tree.root.right.value);
            Assert.IsNull(tree.root.right.right);
            Assert.IsNull(tree.root.right.left);
        }

        [TestCase(5, 5, 2, 2, 4, 4, 3, 3)]
        public void Delete_RootInCoarseTree_UpdateRootFromLeftNeighbor(int rootKey, int rootValue, int leftKey,
            int leftValue, int rightOfLeftKey, int rightOfLeftValue, int leftOfRightOfLeftKey,
            int leftOfRightOfLeftValue)
        {
            var tree = new BSTCoarseGrained<int>();
            tree.Insert(rootKey, rootValue);
            tree.Insert(leftKey, leftValue);
            tree.Insert(rightOfLeftKey, rightOfLeftValue);
            tree.Insert(leftOfRightOfLeftKey, leftOfRightOfLeftValue);

            tree.Delete(rootKey);

            Assert.AreEqual(rightOfLeftValue, tree.root.value);
            Assert.IsNull(tree.root.right);
            Assert.IsNotNull(tree.root.left);

            Assert.AreEqual(leftValue, tree.root.left.value);
            Assert.IsNull(tree.root.left.left);
            Assert.IsNotNull(tree.root.left.right);

            Assert.AreEqual(leftOfRightOfLeftValue, tree.root.left.right.value);
            Assert.IsNull(tree.root.left.right.left);
            Assert.IsNull(tree.root.left.right.right);
        }

        [TestCase(5, 5, 8, 8, 6, 6, 7, 7)]
        public void Delete_RootInCoarseTree_UpdateRootFromRightNeighbor(int rootKey, int rootValue, int rightKey,
            int rightValue, int leftOrRightKey, int leftOfRightValue, int rightOfLeftOfOfRightKey,
            int rightOfLeftOfOfRightValue)
        {
            var tree = new BSTCoarseGrained<int>();
            tree.Insert(rootKey, rootValue);
            tree.Insert(rightKey, rightValue);
            tree.Insert(leftOrRightKey, leftOfRightValue);
            tree.Insert(rightOfLeftOfOfRightKey, rightOfLeftOfOfRightValue);

            tree.Delete(rootKey);

            Assert.AreEqual(leftOfRightValue, tree.root.value);
            Assert.IsNull(tree.root.left);
            Assert.IsNotNull(tree.root.right);

            Assert.AreEqual(rightValue, tree.root.right.value);
            Assert.IsNull(tree.root.right.right);
            Assert.IsNotNull(tree.root.right.left);

            Assert.AreEqual(rightOfLeftOfOfRightValue, tree.root.right.left.value);
            Assert.IsNull(tree.root.right.left.left);
            Assert.IsNull(tree.root.right.left.right);
        }

        [TestCase(5, 5, 8, 8, 7, 7, 6, 6)]
        public void Delete_InnerNodeInCoarseTree_UpdateFromRight(int rootKey, int rootValue, int rightKey,
            int rightValue, int leftOrRightKey, int leftOfRightValue, int leftOfLeftOfRightKey,
            int leftOfLeftOfRightValue)
        {
            var tree = new BSTCoarseGrained<int>();
            tree.Insert(rootKey, rootValue);
            tree.Insert(rightKey, rightValue);
            tree.Insert(leftOrRightKey, leftOfRightValue);
            tree.Insert(leftOfLeftOfRightKey, leftOfLeftOfRightValue);

            tree.Delete(rightKey);

            Assert.AreEqual(rootValue, tree.root.value);
            Assert.IsNull(tree.root.left);
            Assert.IsNotNull(tree.root.right);

            Assert.AreEqual(leftOfRightValue, tree.root.right.value);
            Assert.IsNull(tree.root.right.right);
            Assert.IsNotNull(tree.root.right.left);

            Assert.AreEqual(leftOfLeftOfRightValue, tree.root.right.left.value);
            Assert.IsNull(tree.root.right.left.left);
            Assert.IsNull(tree.root.right.left.right);
        }

        [TestCase(5, 5, 1, 1, 3, 3, 4, 4)]
        public void Delete_InnerNodeInCoarseTree_UpdateFromLeft(int rootKey, int rootValue, int leftKey,
            int leftValue, int rightOfLeftKey, int rightOfLeftValue, int rightOfRightOfLeftKey,
            int rightOfRightOfLeftValue)
        {
            var tree = new BSTCoarseGrained<int>();
            tree.Insert(rootKey, rootValue);
            tree.Insert(leftKey, leftValue);
            tree.Insert(rightOfLeftKey, rightOfLeftValue);
            tree.Insert(rightOfRightOfLeftKey, rightOfRightOfLeftValue);

            tree.Delete(leftKey);

            Assert.AreEqual(rootValue, tree.root.value);
            Assert.IsNull(tree.root.right);
            Assert.IsNotNull(tree.root.left);

            Assert.AreEqual(rightOfLeftValue, tree.root.left.value);
            Assert.IsNull(tree.root.left.left);
            Assert.IsNotNull(tree.root.left.right);

            Assert.AreEqual(rightOfRightOfLeftValue, tree.root.left.right.value);
            Assert.IsNull(tree.root.left.right.left);
            Assert.IsNull(tree.root.left.right.right);
        }

        [TestCase(1, 1, 2, 2)]
        public void Delete_RightLeafInCoarseTree_UpdateTree(int rootKey, int rootValue, int rightKey, int rightValue)
        {
            var tree = new BSTCoarseGrained<int>();
            tree.Insert(rootKey, rootValue);
            tree.Insert(rightKey, rightValue);

            tree.Delete(rightKey);

            Assert.AreEqual(rootValue, tree.root.value);
            Assert.IsNull(tree.root.left);
            Assert.IsNull(tree.root.right);
        }

        [TestCase(2, 2, 1, 1)]
        public void Delete_LeftLeafInCoarseTree_UpdateTree(int rootKey, int rootValue, int leftKey, int leftValue)
        {
            var tree = new BSTCoarseGrained<int>();
            tree.Insert(rootKey, rootValue);
            tree.Insert(leftKey, leftValue);

            tree.Delete(leftKey);

            Assert.AreEqual(rootValue, tree.root.value);
            Assert.IsNull(tree.root.left);
            Assert.IsNull(tree.root.right);
        }
    }
}