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

        protected Boolean IsValid(CoarsePlace place)
        {
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
            if (!mutex.WaitOne(0))
            {
                return false;
            }
            mutex.ReleaseMutex();

            return IsValid(Root);
        }
    }
}
