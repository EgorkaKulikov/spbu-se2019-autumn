using System;

namespace Task05
{
    class RoughlySynchronizedBinarySearchTree<K, V> : Tree<K, V, BinarySearchNode<K, V>>
        where K : IComparable<K>, new()
        where V : struct
    {
        public RoughlySynchronizedBinarySearchTree(params (K, V)[] elements) {
            overRoot = new BinarySearchNode<K, V>(new K(), new V());
            insert(elements);
        }

        protected override BinarySearchNode<K, V> createRoot(K key, V value) => new BinarySearchNode<K, V>(key, value);

        protected override NodeStatus setWithInfo(K key, V? value)
        {
            overRoot.mutex.WaitOne();
            if (root == null)
            {
                if (value == null)
                {
                    overRoot.mutex.ReleaseMutex();
                    return NodeStatus.NodeNotFounded;
                }
                else
                {
                    root = createRoot(key, value.Value);
                    overRoot.mutex.ReleaseMutex();
                    return NodeStatus.NodeCreate;
                }
            }
            BinarySearchNode<K, V> node = root, nextNode;
            while ((nextNode = node.nextNode(key)) != null)
                node = nextNode;
            if (value != null)
            {
                if (node.key.CompareTo(key) == 0)
                {
                    node.value = value.Value;
                    overRoot.mutex.ReleaseMutex();
                    return NodeStatus.NodeUpdate;
                }
                node.createSon(key, value.Value, node.findKeyType(key));
                overRoot.mutex.ReleaseMutex();
                return NodeStatus.NodeCreate;
            }
            if (node.key.CompareTo(key) != 0)
            {
                overRoot.mutex.ReleaseMutex();
                return NodeStatus.NodeNotFounded;
            }
            if (node.left == null && node.right == null)
            {
                if (node.type == SonType.Root)
                    root = null;
                else
                    node.parent.setSon(null, node.type);
                overRoot.mutex.ReleaseMutex();
                return NodeStatus.NodeDelete;
            }
            if (node.right == null)
            {
                if (node.type == SonType.Root)
                    root = node.left;
                else
                    node.left.setFather(node.parent, node.type);
                overRoot.mutex.ReleaseMutex();
                return NodeStatus.NodeDelete;
            }
            BinarySearchNode<K, V> nodeWithNextKey = node.right;
            while (nodeWithNextKey.left != null)
                nodeWithNextKey = nodeWithNextKey.left;
            if (node.type == SonType.Root)
                root = nodeWithNextKey;
            nodeWithNextKey.parent.setSon(nodeWithNextKey.right, nodeWithNextKey.type);
            nodeWithNextKey.moveOn(node);
            overRoot.mutex.ReleaseMutex();
            return NodeStatus.NodeDelete;
        }
    }
}
