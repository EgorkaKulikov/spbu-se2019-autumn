using System;

using System.Threading;

namespace Task05
{
    public class Tree<K, V> : AbstractTree<K, V, Tree<K, V>.Place> where K : IComparable<K>
    {
        public class Place : NodePlace
        {
        }

        private readonly Mutex mutex = new Mutex();

        protected override Place Root { get; } = new Place();
        protected override Place CreatePlace() => new Place();

        protected override Place FindPlace(K key)
        {
            mutex.WaitOne();

            return FindRecursive(Root);

            Place FindRecursive(Place current)
            {
                if (current.node == null) return current;

                var comparisonResult = key.CompareTo(current.node.key);

                if (comparisonResult < 0) return FindRecursive(current.node.left);
                if (comparisonResult > 0) return FindRecursive(current.node.right);

                return current;
            }
        }

        protected override void ReleasePlace(Place place)
        {
            mutex.ReleaseMutex();
        }

        protected override V DeleteRootOf(Place subtree)
        {
            var oldRoot = subtree.node;

            if (subtree.node.right.node == null)
            {
                subtree.node = subtree.node.left.node;
            }
            else
            {
                var newRoot = subtree.node.right;

                while (newRoot.node.left.node != null)
                {
                    newRoot = newRoot.node.left;
                }

                subtree.node = newRoot.node;
                newRoot.node = subtree.node.right.node;
                subtree.node.right.node = oldRoot.right.node;
            }

            return oldRoot.value;
        }
    }
}
