using System;
using System.Threading;

namespace Task05
{
    public class FineTree<K, V> : AbstractTree<K, V, FineTree<K, V>.FineNodePlace> where K : IComparable<K>
    {
        public class FineNodePlace : NodePlace
        {
            public Mutex nodeLock = new Mutex();
        }

        protected override FineNodePlace Root { get; } = new FineNodePlace();

        protected override FineNodePlace CreatePlace() => new FineNodePlace();

        protected override FineNodePlace FindPlace(K key)
        {
            Root.nodeLock.WaitOne();

            return FindRecursive(Root);

            FineNodePlace FindRecursive(FineNodePlace current)
            {
                if (current.node == null) return current;

                var comparisonResult = key.CompareTo(current.node.key);
                FineNodePlace next;

                if (comparisonResult < 0)
                {
                    next = current.node.left;
                }
                else if (comparisonResult > 0)
                {
                    next = current.node.right;
                }
                else
                {
                    return current;
                }

                next.nodeLock.WaitOne();
                current.nodeLock.ReleaseMutex();
                return FindRecursive(next);
            }
        }

        protected override void ReleasePlace(FineNodePlace place)
        {
            place.nodeLock.ReleaseMutex();
        }

        private void Sync(FineNodePlace place) {
            place.nodeLock.WaitOne();
            place.nodeLock.ReleaseMutex();
        }

        protected override V DeleteRootOf(FineNodePlace subtree)
        {
            var oldRoot = subtree.node;

            Sync(oldRoot.right);

            if (oldRoot.right.node == null)
            {
                Sync(oldRoot.left);
                subtree.node = oldRoot.left.node;
            }
            else
            {
                var newRoot = oldRoot.right;
                Sync(newRoot.node.left);

                while (newRoot.node.left.node != null)
                {
                    newRoot = newRoot.node.left;
                    Sync(newRoot.node.left);
                }

                subtree.node = newRoot.node;
                Sync(newRoot.node.right);
                newRoot.node = newRoot.node.right.node;
                subtree.node.right.node = oldRoot.right.node;
            }

            return oldRoot.value;
        }
    }
}