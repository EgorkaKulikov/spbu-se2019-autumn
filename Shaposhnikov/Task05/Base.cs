namespace Task05
{
    /*public class bases
    {
        private Node Root { get; set; }
    
        public void Insert(int value)
        {
            Node prev = null, current = this.Root;    

            while (current != null)
            {
                prev = current;
                if (value < current.Value)
                    current = current.Left;
                else if (value > current.Value)
                    current = current.Right;
                else
                    return; //already exists
            }

            Node newNode = new Node {Value = value};

            if (this.Root == null)
                this.Root = newNode;
            else
            {
                if (value < prev.Value)
                    prev.Left = newNode;
                else
                    prev.Right = newNode;
            }
        }

        public Node Find(int value)
        {
            return this.Find(value, this.Root);
        }

        public void Remove(int key)
        {
            Remove(this.Root, key);
        }

        private Node Find(int value, Node parent)
        {
            if (parent != null)
            {
                if (value == parent.Value) return parent;
                if (value < parent.Value)
                    return Find(value, parent.Left);
                else
                    return Find(value, parent.Right);
            }

            return null;
        }

        private Node Remove(Node parent, int key)
        {
            if (parent == null) return parent;

            //laddering the tree for sought-for node
            if (key < parent.Value)
                parent.Left = Remove(parent.Left, key);
            else if (key > parent.Value)
                parent.Right = Remove(parent.Right, key);
            //the remaining case is parent's value equals key
            else
            {
                //case node with one or zero children
                if (parent.Right == null)
                    return parent.Left;
                else if (parent.Left == null)
                    return parent.Right;
                
                // get the next lowest value after parent's
                parent.Value = MinValue(parent.Right);

                // and then delete the node with this lowest key
                parent.Right = Remove(parent.Right, parent.Value);
            }

            return parent;
        }

        private int MinValue(Node node)
        {
            int min = node.Value;

            while (node.Left != null)
            {
                min = node.Left.Value;
                node = node.Left;
            }

            return min;
        }
    }
*/
}