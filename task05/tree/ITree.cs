namespace Task05
{
    public interface ITree<K, V>
    {
        V Find(K key);
        V Add(K key, V value);
    }
}