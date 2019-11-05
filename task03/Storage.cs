using System.Collections.Generic;
using System.Threading;

namespace Task03 {
    public interface IAccessPolicy {
        void AcquireRead();
        void ReleaseRead();
        void AcquireWrite();
        void ReleaseWrite();
    }

    public class ListReader<T> {
        private readonly List<T> data;

        public int Count {
            get {
                return data.Count;
            }
        }

        public ListReader(List<T> data) {
            this.data = data;
        }

        public T Read(int index) {
            return data[index];
        }
    }

    public class Storage<T> {

        public interface IConsumer {
            void DoConsuming(ListReader<T> data);
        }

        public interface IProducer {
            void DoProducing(List<T> data);
        }

        public void AddConsumer(IConsumer consumer) {
            policy.AcquireRead();
            consumer?.DoConsuming(new ListReader<T>(data));
            policy.ReleaseRead();
        }

        public void AddProducer(IProducer producer) {
            policy.AcquireWrite();
            producer?.DoProducing(data);
            policy.ReleaseWrite();
        }

        private readonly List<T> data = new List<T>();
        private readonly IAccessPolicy policy;

        public Storage(IAccessPolicy policy) {
            this.policy = policy;
        }
    }
}
