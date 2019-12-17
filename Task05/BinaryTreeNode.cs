using System;
using System.Threading;

namespace Task05
{
    public class BinaryTreeNode : IEquatable<BinaryTreeNode>
    {
        public readonly int Key;
        public BinaryTreeNode Left;
        public BinaryTreeNode Parent;
        public BinaryTreeNode Right;
        public readonly Mutex NodeMutex = new Mutex();

        public BinaryTreeNode(int key)
        {
            Key = key;
        }

        public bool Equals(BinaryTreeNode other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            return Key == other.Key;
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((BinaryTreeNode) obj);
        }

        public override int GetHashCode()
        {
            return Key;
        }

        public static bool operator ==(BinaryTreeNode left, BinaryTreeNode right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(BinaryTreeNode left, BinaryTreeNode right)
        {
            return !Equals(left, right);
        }
    }
}