using System;

namespace Task05
{
    public interface ITree<K, V>
    {
        V Add(K key, V value);
        V Find(K key);
        V Delete(K key);
        Boolean IsValid();        
    }

    public abstract class AbstractTree<K, V, NP>: ITree<K, V> where NP : AbstractTree<K, V, NP>.NodePlace
    {
        public class NodePlace
        {
            public Node node = null;
        }

        public class Node
        {
            public K key;
            public V value;
            public readonly NP left;
            public readonly NP right;

            public Node(K key, V value, NP left, NP right)
            {
                this.key = key;
                this.value = value;
                this.left = left;
                this.right = right;
            }
        }

        protected abstract NP Root { get; }
        protected abstract NP CreatePlace();
        protected abstract NP FindPlace(K key);
        protected abstract V DeleteRootOf(NP place);
        protected abstract void ReleasePlace(NP place);
        public abstract Boolean IsValid();

        public V Find(K key)
        {
            var place = FindPlace(key);

            V result = default;
            if (place.node != null)
            {
                result = place.node.value;
            }

            ReleasePlace(place);
            return result;
        }

        public V Add(K key, V value)
        {
            var place = FindPlace(key);

            V result = default;
            if (place.node == null)
            {
                place.node = new Node(key, value, CreatePlace(), CreatePlace());
            }
            else
            {
                result = place.node.value;
            }

            ReleasePlace(place);
            return result;
        }

        public V Delete(K key)
        {
            var place = FindPlace(key);

            V result = default;
            if (place.node != null)
            {
                result = DeleteRootOf(place);
            }

            ReleasePlace(place);
            return result;
        }
    }
}
