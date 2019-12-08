using System;
using System.Threading;

namespace Task05
{
    public class FineTree<K, V> : AbstractTree<K, V, FineTree<K, V>.FinePlace> where K : IComparable<K>
    {
        public class FinePlace : NodePlace
        {
            public Mutex nodeLock = new Mutex();
        }

        protected override FinePlace Root { get; } = new FinePlace();

        protected override FinePlace CreatePlace() => new FinePlace();

        protected override FinePlace FindPlace(K key)
        {
            Root.nodeLock.WaitOne();

            return FindRecursive(Root);

            FinePlace FindRecursive(FinePlace current)
            {
                if (current.node == null) return current;

                var comparisonResult = key.CompareTo(current.node.key);
                FinePlace next;

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

        protected override void ReleasePlace(FinePlace place)
        {
            place.nodeLock.ReleaseMutex();
        }

        private void Sync(FinePlace place) {
            place.nodeLock.WaitOne();
            place.nodeLock.ReleaseMutex();
        }

        protected override V DeleteRootOf(FinePlace subtree)
        {
            var oldRoot = subtree.node;

            Sync(oldRoot.right);
            Sync(oldRoot.left);

            if (oldRoot.right.node == null)
            {
                subtree.node = oldRoot.left.node;
            } else if (oldRoot.left.node == null)
            {
                subtree.node = oldRoot.right.node;
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

                Sync(newRoot.node.right);
                subtree.node = newRoot.node;
                newRoot.node = newRoot.node.right.node;
                subtree.node.right.node = oldRoot.right.node;
                subtree.node.left.node = oldRoot.left.node;
            }

            return oldRoot.value;
        }
    }
}
