using System;
using System.Threading;

namespace Task05
{
    public class FineTree<K, V> : AbstractTree<K, V> where K : IComparable<K>
    {
        public override V Find(K key)
        {
            Lock(rootLock);

            V result = default;
            var current = root;
            Lock(current);

            Unlock(rootLock);

            while (current != null)
            {
                Node next = null;

                var comparisonResult = key.CompareTo(current.key);

                if (comparisonResult < 0)
                {
                    next = current.left;
                }
                else if (comparisonResult > 0)
                {
                    next = current.right;
                }
                else
                {
                    result = current.value;
                }

                Lock(next);
                Unlock(current);
                current = next;
            }

            return result;
        }

        public override V Add(K key, V value)
        {
            Lock(rootLock);

            V result = default;
            var current = root;
            Lock(current);

            if (root == null)
            {
                root = new Node(key, value);
            }

            Unlock(rootLock);

            while (current != null)
            {
                var next = current;

                var comparisonResult = key.CompareTo(current.key);

                if (comparisonResult < 0)
                {
                    next = current.left;
                    if (next == null)
                    {
                        current.left = new Node(key, value);
                    }
                }
                else if (comparisonResult > 0)
                {
                    next = current.right;
                    if (next == null)
                    {
                        current.right = new Node(key, value);
                    }
                }
                else
                {
                    result = current.value;
                    current.value = value;
                }

                Lock(next);
                Unlock(current);
                current = next;
            }

            return result;
        }
    }
}
