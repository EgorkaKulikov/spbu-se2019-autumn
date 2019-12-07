using System;
using System.Threading;

namespace Task05
{
    public class CoarseTree<K, V> : AbstractTree<K, V, CoarseTree<K, V>.CoarsePlace> where K : IComparable<K>
    {
        public class CoarsePlace : NodePlace
        {
        }

        private readonly Mutex mutex = new Mutex();

        protected override CoarsePlace Root { get; } = new CoarsePlace();
        protected override CoarsePlace CreatePlace() => new CoarsePlace();

        protected override CoarsePlace FindPlace(K key)
        {
            mutex.WaitOne();

            return FindRecursive(Root);

            CoarsePlace FindRecursive(CoarsePlace current)
            {
                if (current.node == null) return current;

                var comparisonResult = key.CompareTo(current.node.key);

                if (comparisonResult < 0) return FindRecursive(current.node.left);
                if (comparisonResult > 0) return FindRecursive(current.node.right);

                return current;
            }
        }

        protected override void ReleasePlace(CoarsePlace place)
        {
            mutex.ReleaseMutex();
        }

        protected override V DeleteRootOf(CoarsePlace subtree)
        {
            var oldRoot = subtree.node;

            if (oldRoot.right.node == null)
            {
                subtree.node = oldRoot.left.node;
            }
            else if (oldRoot.left.node == null)
            {
                subtree.node = oldRoot.right.node;
            }
            else
            {
                var newRoot = oldRoot.right;

                while (newRoot.node.left.node != null)
                {
                    newRoot = newRoot.node.left;
                }

                subtree.node = newRoot.node;
                newRoot.node = newRoot.node.right.node;
                subtree.node.right.node = oldRoot.right.node;
                subtree.node.left.node = oldRoot.left.node;
            }

            return oldRoot.value;
        }
    }
}
