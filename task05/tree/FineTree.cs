using System;
using System.Threading;

namespace Task05
{
    public class FineTree<K, V> : AbstractTree<K, V, FineTree<K, V>.FinePlace> where K : IComparable<K>
    {
        public class FinePlace : NodePlace
        {
            public SemaphoreSlim nodeLock = new SemaphoreSlim(1, 1);
        }

        protected override FinePlace Root { get; } = new FinePlace();

        protected override FinePlace CreatePlace() => new FinePlace();

        protected override FinePlace FindPlace(K key)
        {
            Root.nodeLock.Wait();

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

                next.nodeLock.Wait();
                current.nodeLock.Release();
                return FindRecursive(next);
            }
        }

        protected override void ReleasePlace(FinePlace place)
        {
            place.nodeLock.Release();
        }

        private void Sync(FinePlace place) {
            place.nodeLock.Wait();
            place.nodeLock.Release();
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

        protected Boolean IsValid(FinePlace place)
        {
            if (!place.nodeLock.Wait(0))
            {
                return false;
            }
            place.nodeLock.Release();


            if (place.node == null)
            {
                return true;
            }

            if (place.node.left.node != null
                &&
                place.node.left.node.key.CompareTo(place.node.key) >= 0)
            {
                return false;
            }

            if (place.node.right.node != null
                &&
                place.node.right.node.key.CompareTo(place.node.key) <= 0)
            {
                return false;
            }

            return IsValid(place.node.left) && IsValid(place.node.right);
        }

        public override Boolean IsValid()
        {
            return IsValid(Root);
        }
    }
}
