namespace Task05
{
    public abstract class AbstractNode<T>
    {
        public int key;
        public T value;
        public AbstractNode<T> left;
        public AbstractNode<T> right;
        public AbstractNode<T> parent;
      //  public abstract AbstractNode(AbstractNode<T> parent, int key, T value);
    }
    public class Node<T> : AbstractNode<T>
    {
        public int key;
        public T value;
        public Node<T> left;
        public Node<T> right;
        public Node<T> parent;
        public Node (Node<T> parent, int key, T value)
        {
            this.key = key;
            this.value = value;
            left = null;
            right = null;
            this.parent = parent;
        }
    }
}